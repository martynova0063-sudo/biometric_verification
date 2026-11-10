import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from jose import jwt

from app.authentication import (
    JWT_ACCESS_TTL_SECONDS,
    JWT_ISSUER,
    JWT_PASSWORD_SUBJECT,
    load_bearer_user,
    password_login,
)
from app.main import app
from app.schemas import PasswordLoginInput


class PasswordJwtAuthenticationTests(unittest.TestCase):
    password = "test-only-password"
    signing_secret = "unit-test-signing-key-that-is-long-enough-32-bytes"

    def test_correct_password_returns_a_short_lived_bearer_token(self):
        with patch.dict(
            os.environ,
            {
                "AUTH_LOGIN_PASSWORD": self.password,
                "AUTH_JWT_SECRET": self.signing_secret,
            },
        ):
            result = password_login(PasswordLoginInput(password=self.password))
            claims = jwt.decode(
                result.access_token,
                self.signing_secret,
                algorithms=["HS256"],
                issuer=JWT_ISSUER,
            )

            user = load_bearer_user(f"Bearer {result.access_token}")

        self.assertEqual(result.token_type, "bearer")
        self.assertEqual(result.expires_in, JWT_ACCESS_TTL_SECONDS)
        self.assertEqual(claims["sub"], JWT_PASSWORD_SUBJECT)
        self.assertEqual(claims["exp"] - claims["iat"], JWT_ACCESS_TTL_SECONDS)
        self.assertEqual(user["id"], JWT_PASSWORD_SUBJECT)

    def test_incorrect_password_is_rejected(self):
        with patch.dict(
            os.environ,
            {
                "AUTH_LOGIN_PASSWORD": self.password,
                "AUTH_JWT_SECRET": self.signing_secret,
            },
        ):
            with self.assertRaises(HTTPException) as raised:
                password_login(PasswordLoginInput(password="incorrect"))

        self.assertEqual(raised.exception.status_code, 401)

    def test_missing_or_weak_configuration_disables_password_login(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HTTPException) as missing:
                password_login(PasswordLoginInput(password=self.password))

        self.assertEqual(missing.exception.status_code, 503)

        with patch.dict(
            os.environ,
            {
                "AUTH_LOGIN_PASSWORD": self.password,
                "AUTH_JWT_SECRET": "too-short",
            },
        ):
            with self.assertRaises(HTTPException) as weak:
                password_login(PasswordLoginInput(password=self.password))

        self.assertEqual(weak.exception.status_code, 503)

    def test_invalid_bearer_token_does_not_authenticate(self):
        with patch.dict(
            os.environ,
            {"AUTH_JWT_SECRET": self.signing_secret},
        ):
            self.assertIsNone(load_bearer_user("Bearer not-a-jwt"))

    def test_login_endpoint_token_authorizes_protected_api_route(self):
        client = TestClient(app)
        with patch.dict(
            os.environ,
            {
                "AUTH_LOGIN_PASSWORD": self.password,
                "AUTH_JWT_SECRET": self.signing_secret,
            },
        ):
            bad_login = client.post(
                "/api/auth/login",
                json={"password": "incorrect"},
            )
            login = client.post(
                "/api/auth/login",
                json={"password": self.password},
            )
            protected_without_token = client.get("/api/arcface/status")
            protected_with_token = client.get(
                "/api/arcface/status",
                headers={"Authorization": f"Bearer {login.json()['access_token']}"},
            )

        self.assertEqual(bad_login.status_code, 401)
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json()["token_type"], "bearer")
        self.assertEqual(login.json()["expires_in"], JWT_ACCESS_TTL_SECONDS)
        self.assertEqual(protected_without_token.status_code, 401)
        self.assertEqual(protected_with_token.status_code, 200)


if __name__ == "__main__":
    unittest.main()
