from datetime import datetime, timedelta, timezone
import hmac
import hashlib
import logging
import os
import secrets
from urllib.parse import urlencode, urlsplit

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from jose import JWTError, jwt
from sqlalchemy import delete
from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import AuthSession
from .schemas import PasswordLoginInput, PasswordLoginToken


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
ISSUER_URL = os.getenv("ISSUER_URL", "https://replit.com/oidc").rstrip("/")
SESSION_COOKIE = "sid"
SESSION_TTL = timedelta(days=7)
JWT_ISSUER = "ukz-health-id"
JWT_PASSWORD_SUBJECT = "password-api-user"
JWT_ACCESS_TTL_SECONDS = 900
OIDC_ISSUER_METADATA = f"{ISSUER_URL}/.well-known/openid-configuration"
oauth = OAuth()
oauth.register(
    name="replit",
    client_id=os.getenv("REPL_ID"),
    server_metadata_url=OIDC_ISSUER_METADATA,
    client_kwargs={
        "scope": "openid email profile offline_access",
        "code_challenge_method": "S256",
    },
)


def session_hash(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def get_oidc_client():
    if not os.getenv("REPL_ID"):
        raise HTTPException(status_code=503, detail="Вход в систему не настроен.")
    client = oauth.create_client("replit")
    if client is None:
        raise HTTPException(status_code=503, detail="Сервис входа временно недоступен.")
    return client


def safe_return_to(value: str | None) -> str:
    if (
        not value
        or "\\" in value
        or any(ord(char) < 32 for char in value)
        or not value.startswith("/")
        or value.startswith("//")
    ):
        return "/"
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/") or parsed.path.startswith("//"):
        return "/"
    return value


def public_origin(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", "https").split(",", 1)[0].strip()
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    ).split(",", 1)[0].strip()
    if proto not in {"http", "https"} or not host or any(char in host for char in "\r\n/@"):
        raise HTTPException(status_code=400, detail="Недопустимый адрес запроса.")
    return f"{proto}://{host}"


def current_user(request: Request) -> dict | None:
    return getattr(request.state, "auth_user", None)


def load_bearer_user(authorization: str | None) -> dict | None:
    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token.strip():
        return None

    signing_secret = os.getenv("AUTH_JWT_SECRET")
    if not signing_secret or len(signing_secret.encode("utf-8")) < 32:
        return None
    try:
        claims = jwt.decode(
            token.strip(),
            signing_secret,
            algorithms=["HS256"],
            issuer=JWT_ISSUER,
            options={
                "require_sub": True,
                "require_exp": True,
                "require_iat": True,
                "require_iss": True,
            },
        )
    except (JWTError, TypeError, ValueError):
        return None
    if claims.get("sub") != JWT_PASSWORD_SUBJECT:
        return None

    return {
        "id": JWT_PASSWORD_SUBJECT,
        "email": None,
        "firstName": "API",
        "lastName": "Password",
        "profileImageUrl": None,
    }


def load_session_user(db: Session, session_id: str | None) -> dict | None:
    if not session_id:
        return None
    session = db.get(AuthSession, session_hash(session_id))
    if session is None:
        return None
    now = datetime.utcnow()
    if session.expires_at <= now:
        db.delete(session)
        db.commit()
        return None
    return {
        "id": session.user_id,
        "email": session.email,
        "firstName": session.first_name,
        "lastName": session.last_name,
        "profileImageUrl": session.profile_image_url,
    }


def store_auth_session(claims: dict) -> str:
    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("Identity provider did not return a subject.")

    session_id = secrets.token_urlsafe(48)
    record = AuthSession(
        sid_hash=session_hash(session_id),
        user_id=user_id,
        email=_claim_string(claims.get("email"), 320),
        first_name=_claim_string(claims.get("first_name") or claims.get("given_name"), 160),
        last_name=_claim_string(claims.get("last_name") or claims.get("family_name"), 160),
        profile_image_url=_claim_string(
            claims.get("profile_image_url") or claims.get("picture"), 2000
        ),
        expires_at=datetime.utcnow() + SESSION_TTL,
    )
    with SessionLocal() as db:
        db.add(record)
        db.commit()
    return session_id


def _claim_string(value: object, max_length: int) -> str | None:
    return value[:max_length] if isinstance(value, str) and value else None


def set_session_cookie(response, session_id: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )


@router.post("/auth/login", response_model=PasswordLoginToken)
def password_login(payload: PasswordLoginInput) -> PasswordLoginToken:
    expected_password = os.getenv("AUTH_LOGIN_PASSWORD")
    signing_secret = os.getenv("AUTH_JWT_SECRET")
    if not expected_password:
        raise HTTPException(
            status_code=503,
            detail="Секрет AUTH_LOGIN_PASSWORD не настроен.",
        )
    if not signing_secret:
        raise HTTPException(
            status_code=503,
            detail="Секрет AUTH_JWT_SECRET не настроен.",
        )
    if len(signing_secret.encode("utf-8")) < 32:
        raise HTTPException(
            status_code=503,
            detail="AUTH_JWT_SECRET должен содержать не менее 32 байт.",
        )

    if not hmac.compare_digest(
        payload.password.encode("utf-8"),
        expected_password.encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Неверный пароль.")

    issued_at = int(datetime.now(timezone.utc).timestamp())
    expires_at = issued_at + JWT_ACCESS_TTL_SECONDS
    access_token = jwt.encode(
        {
            "sub": JWT_PASSWORD_SUBJECT,
            "iss": JWT_ISSUER,
            "iat": issued_at,
            "exp": expires_at,
        },
        signing_secret,
        algorithm="HS256",
    )
    return PasswordLoginToken(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_ACCESS_TTL_SECONDS,
    )


@router.get("/auth/user")
async def get_current_auth_user(request: Request):
    return {"user": current_user(request)}


@router.get("/login")
async def begin_login(request: Request, return_to: str | None = Query(None, alias="returnTo")):
    client = get_oidc_client()
    request.session["return_to"] = safe_return_to(return_to)
    callback_url = f"{public_origin(request)}/api/callback"
    return await client.authorize_redirect(
        request,
        callback_url,
        prompt="login consent",
        code_challenge_method="S256",
    )


@router.get("/callback")
async def finish_login(request: Request):
    client = get_oidc_client()
    return_to = safe_return_to(request.session.get("return_to"))
    callback_url = f"{public_origin(request)}/api/callback"
    try:
        token = await client.authorize_access_token(request)
        claims = token.get("userinfo")
        if not claims:
            claims = await client.userinfo(token=token)
        if not claims:
            raise ValueError("No verified user claims were returned.")
        session_id = store_auth_session(dict(claims))
    except Exception as error:
        # Do not log authorization codes, tokens, or callback query strings.
        logger.warning("OIDC sign-in failed (%s).", type(error).__name__)
        request.session.clear()
        return RedirectResponse("/?login_error=1", status_code=303)

    request.session.clear()
    response = RedirectResponse(return_to, status_code=303)
    set_session_cookie(response, session_id)
    return response


@router.get("/logout")
async def logout(request: Request, return_to: str | None = Query(None, alias="returnTo")):
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        with SessionLocal() as db:
            db.execute(delete(AuthSession).where(AuthSession.sid_hash == session_hash(session_id)))
            db.commit()

    destination = safe_return_to(return_to)
    response = RedirectResponse(destination, status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    request.session.clear()

    try:
        client = get_oidc_client()
        metadata = await client.load_server_metadata()
        end_session_endpoint = metadata.get("end_session_endpoint")
        if end_session_endpoint:
            logout_url = (
                f"{end_session_endpoint}?"
                f"{urlencode({'client_id': os.getenv('REPL_ID', ''), 'post_logout_redirect_uri': f'{public_origin(request)}{destination}'})}"
            )
            response = RedirectResponse(logout_url, status_code=303)
            response.delete_cookie(
                SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="lax"
            )
    except Exception:
        logger.info("OIDC provider logout endpoint is unavailable; local session was cleared.")
    return response


def auth_required_response() -> JSONResponse:
    return JSONResponse({"detail": "Authentication required"}, status_code=401)
