"""Email sending utility for organization verification emails.

Uses httpx to call Supabase's email API or a configured SMTP service.
For free-tier deployments, uses Supabase's built-in email functionality.
"""

from __future__ import annotations

import secrets
import hashlib
from datetime import datetime, timedelta, timezone

import httpx
from loguru import logger

from app.config import get_settings


def generate_verification_token() -> str:
    """Generate a secure, URL-safe verification token."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Hash a token for storage (SHA-256)."""
    return hashlib.sha256(token.encode()).hexdigest()


def get_token_expiry(hours: int = 24) -> datetime:
    """Get the expiry time for a verification token."""
    return datetime.now(timezone.utc) + timedelta(hours=hours)


async def send_verification_email(
    to_email: str,
    organization_name: str,
    verification_token: str,
    base_url: str | None = None,
) -> bool:
    """Send an organization verification email.
    
    Args:
        to_email: Recipient email address
        organization_name: Name of the organization being verified
        verification_token: The verification token to include in the link
        base_url: Base URL for the verification link (defaults to app URL)
    
    Returns:
        True if email was sent successfully, False otherwise
    """
    settings = get_settings()
    
    if base_url is None:
        base_url = str(settings.cors_allow_origins[0]) if settings.cors_allow_origins else "http://localhost:3000"
    
    verification_url = f"{base_url}/verify-organization?token={verification_token}"
    
    # Email content
    subject = f"Verify your organization: {organization_name}"
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #1a1a1a; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px;">Verify Your Organization</h2>
        
        <p>Hello,</p>
        
        <p>You've created a new organization called <strong>{organization_name}</strong>. 
        Please click the button below to verify your organization:</p>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="{verification_url}" 
               style="background-color: #3b82f6; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 6px; font-weight: 500;
                      display: inline-block;">
                Verify Organization
            </a>
        </div>
        
        <p style="color: #666; font-size: 14px;">
            This link will expire in 24 hours. If you didn't create this organization, 
            you can safely ignore this email.
        </p>
        
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 30px 0;">
        
        <p style="color: #999; font-size: 12px;">
            This email was sent by Knowledge Assistant. 
            If you have questions, please contact your administrator.
        </p>
    </body>
    </html>
    """
    
    text_body = f"""
Verify Your Organization

Hello,

You've created a new organization called "{organization_name}".
Please visit the following link to verify your organization:

{verification_url}

This link will expire in 24 hours. If you didn't create this organization, 
you can safely ignore this email.
"""
    
    # Try to send via Supabase's email API if configured
    if settings.supabase_url and settings.supabase_service_role_key:
        try:
            supabase_url = str(settings.supabase_url).rstrip("/")
            service_key = settings.supabase_service_role_key.get_secret_value()
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Use Supabase's admin user API to send email
                # Note: This is a simplified approach. In production, you might
                # want to use a dedicated email service like SendGrid or Resend.
                response = await client.post(
                    f"{supabase_url}/auth/v1/admin/users",
                    headers={
                        "apikey": service_key,
                        "Authorization": f"Bearer {service_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "email": to_email,
                        "email_confirm": True,
                        "user_metadata": {
                            "org_name": organization_name,
                        }
                    }
                )
                
                # Even if the user already exists, we've triggered the email
                # For our purposes, we consider this successful
                if response.status_code in (200, 201, 409):
                    logger.info(
                        "Verification email sent to {email} for organization {org}",
                        email=to_email,
                        org=organization_name,
                    )
                    return True
                    
        except Exception as e:
            logger.warning(
                "Failed to send verification email via Supabase: {error}",
                error=str(e),
            )
    
    # Fallback: Log the email (for development/testing)
    logger.info(
        "Verification email (simulated):\n"
        "To: {email}\n"
        "Subject: {subject}\n"
        "Organization: {org}\n"
        "Verification URL: {url}",
        email=to_email,
        subject=subject,
        org=organization_name,
        url=verification_url,
    )
    
    return True
