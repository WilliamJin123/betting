"""
Polymarket Authentication
=========================

Loads and validates API credentials needed to place orders on Polymarket.

HOW POLYMARKET AUTH WORKS (for beginners):
    Polymarket uses a two-layer authentication system:

    1. **Polygon Wallet Private Key** (POLYMARKET_PRIVATE_KEY):
       This is a cryptographic key that controls your wallet on the Polygon
       blockchain. Think of it like a master password. Polymarket uses this
       to derive your API credentials. NEVER share this with anyone.

    2. **Derived API Credentials** (API key, secret, passphrase):
       When you first connect to Polymarket's CLOB API, your private key is
       used to cryptographically sign a message. Polymarket's server verifies
       that signature and gives you back an API key + secret + passphrase.
       These are like a session token -- they prove you own the wallet without
       sending the private key over the network.

       You can either:
       (a) Let py-clob-client derive them automatically each time (easier), or
       (b) Derive them once, save them as env vars, and reuse them (faster).

HOW TO GET YOUR CREDENTIALS:
    1. Create a Polygon wallet (e.g., via MetaMask).
    2. Fund it with USDC.e (on Polygon network) and a tiny bit of POL for gas.
    3. Export the private key from MetaMask:
       - Open MetaMask -> Account details -> Export private key
       - Store it securely (password manager, not a text file on your desktop)
    4. Set the environment variable:
       export POLYMARKET_PRIVATE_KEY="0xYourPrivateKeyHere"

    Optionally, if you've already derived API creds (from a prior session):
       export POLYMARKET_API_KEY="your-api-key"
       export POLYMARKET_API_SECRET="your-api-secret"
       export POLYMARKET_API_PASSPHRASE="your-api-passphrase"

    5. If using a proxy/smart wallet (advanced), also set:
       export POLYMARKET_FUNDER="0xYourWalletAddress"

SECURITY:
    - NEVER hardcode credentials in source code.
    - NEVER commit a .env file with real keys to version control.
    - Use environment variables or a secrets manager.
    - The private key gives FULL control over your wallet funds.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Credential container
# ---------------------------------------------------------------------------

@dataclass
class PolymarketCredentials:
    """
    Container for all Polymarket authentication credentials.

    Fields:
        private_key:    Polygon wallet private key (hex string, starts with 0x).
                        Required for signing orders.
        api_key:        CLOB API key (derived from private_key).
                        Optional -- py-clob-client can derive it on the fly.
        api_secret:     CLOB API secret (derived from private_key).
                        Optional -- py-clob-client can derive it on the fly.
        api_passphrase: CLOB API passphrase (derived from private_key).
                        Optional -- py-clob-client can derive it on the fly.
        funder:         Wallet address to use as the funder for proxy wallets.
                        Optional -- only needed for proxy/smart contract wallets.
    """
    private_key: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None
    funder: Optional[str] = None

    @property
    def has_derived_creds(self) -> bool:
        """True if all three derived API credentials are set."""
        return all([self.api_key, self.api_secret, self.api_passphrase])


# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

ENV_PRIVATE_KEY = "POLYMARKET_PRIVATE_KEY"
ENV_API_KEY = "POLYMARKET_API_KEY"
ENV_API_SECRET = "POLYMARKET_API_SECRET"
ENV_API_PASSPHRASE = "POLYMARKET_API_PASSPHRASE"
ENV_FUNDER = "POLYMARKET_FUNDER"


# ---------------------------------------------------------------------------
# Load and validate credentials
# ---------------------------------------------------------------------------

def load_credentials() -> PolymarketCredentials:
    """
    Load Polymarket credentials from environment variables.

    The only REQUIRED variable is POLYMARKET_PRIVATE_KEY. The API key,
    secret, and passphrase are optional -- if not set, py-clob-client will
    derive them automatically from the private key (which involves a
    cryptographic handshake with Polymarket's server on first use).

    Returns:
        PolymarketCredentials with all available credentials populated.

    Raises:
        EnvironmentError: If POLYMARKET_PRIVATE_KEY is not set.

    Example:
        # In your shell or .env file:
        # export POLYMARKET_PRIVATE_KEY="0xabc123..."

        from bot.polymarket.auth import load_credentials
        creds = load_credentials()
        print(creds.private_key[:10] + "...")  # "0xabc123..."
    """
    private_key = os.environ.get(ENV_PRIVATE_KEY, "").strip()

    if not private_key:
        _print_setup_help()
        raise EnvironmentError(
            f"Missing required environment variable: {ENV_PRIVATE_KEY}\n"
            f"This is your Polygon wallet private key, needed to sign orders.\n"
            f"Set it with: export {ENV_PRIVATE_KEY}=\"0xYourKeyHere\""
        )

    # Validate format: should be a hex string, optionally starting with 0x
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key
    hex_part = private_key[2:]
    if not all(c in "0123456789abcdefABCDEF" for c in hex_part):
        raise ValueError(
            f"{ENV_PRIVATE_KEY} does not look like a valid hex private key. "
            f"Expected format: 0x followed by 64 hex characters."
        )
    if len(hex_part) != 64:
        logger.warning(
            f"{ENV_PRIVATE_KEY} is {len(hex_part)} hex characters "
            f"(expected 64). This may be invalid."
        )

    # Load optional derived credentials
    api_key = os.environ.get(ENV_API_KEY, "").strip() or None
    api_secret = os.environ.get(ENV_API_SECRET, "").strip() or None
    api_passphrase = os.environ.get(ENV_API_PASSPHRASE, "").strip() or None
    funder = os.environ.get(ENV_FUNDER, "").strip() or None

    creds = PolymarketCredentials(
        private_key=private_key,
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        funder=funder,
    )

    # Log what we found (without leaking secrets)
    logger.info(f"Loaded private key: {private_key[:6]}...{private_key[-4:]}")
    if creds.has_derived_creds:
        logger.info("Using pre-derived API credentials from environment.")
    else:
        logger.info(
            "No pre-derived API creds found. "
            "py-clob-client will derive them from the private key."
        )
    if funder:
        logger.info(f"Funder address: {funder[:6]}...{funder[-4:]}")

    return creds


def validate_credentials(creds: PolymarketCredentials) -> list[str]:
    """
    Check credentials for potential issues without raising exceptions.

    Returns a list of warning messages. Empty list means no issues found.

    This is useful for a pre-flight check before entering live trading mode.
    """
    warnings: list[str] = []

    if not creds.private_key:
        warnings.append("Private key is empty.")
    elif len(creds.private_key) < 66:  # "0x" + 64 hex chars
        warnings.append(
            f"Private key is shorter than expected "
            f"({len(creds.private_key)} chars, expected 66)."
        )

    if not creds.has_derived_creds:
        warnings.append(
            "No pre-derived API credentials. "
            "The client will need to derive them (requires network call)."
        )

    return warnings


# ---------------------------------------------------------------------------
# Helper: print setup instructions when credentials are missing
# ---------------------------------------------------------------------------

def _print_setup_help() -> None:
    """Print user-friendly instructions for setting up credentials."""
    print(
        "\n"
        "=" * 60 + "\n"
        "  POLYMARKET CREDENTIALS NOT FOUND\n"
        "=" * 60 + "\n"
        "\n"
        "  To trade on Polymarket, you need a Polygon wallet private key.\n"
        "\n"
        "  Quick setup:\n"
        "  1. Install MetaMask (browser extension)\n"
        "  2. Switch to the Polygon network\n"
        "  3. Fund your wallet with USDC.e + a tiny bit of POL for gas\n"
        "  4. Export your private key:\n"
        "     MetaMask -> Account -> Export Private Key\n"
        "  5. Set the environment variable:\n"
        "\n"
        f"     export {ENV_PRIVATE_KEY}=\"0xYourPrivateKeyHere\"\n"
        "\n"
        "  Optional (speeds up startup if you've derived them before):\n"
        f"     export {ENV_API_KEY}=\"your-api-key\"\n"
        f"     export {ENV_API_SECRET}=\"your-api-secret\"\n"
        f"     export {ENV_API_PASSPHRASE}=\"your-passphrase\"\n"
        "\n"
        "  SECURITY WARNING:\n"
        "  - NEVER commit your private key to git\n"
        "  - NEVER share it with anyone\n"
        "  - Use a separate wallet with only trading funds\n"
        "  - Consider a .env file (add .env to .gitignore!)\n"
        "\n"
        "=" * 60 + "\n"
    )
