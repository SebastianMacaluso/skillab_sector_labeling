import requests
import logging
from typing import Iterable, Optional, Union, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('integration_external_queries.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

API = "https://skillab-tracker.csd.auth.gr/api"
USERNAME = "skillab_staff"
PASSWORD = "skillroadtrip00"

def get_token() -> str:
    """Authenticate and return a Bearer token."""
    try:
        logger.info(f"Authenticating with API: {API}/login")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD})
        
        if res.status_code != 200:
            logger.error(f"Login failed with status {res.status_code}: {res.text}")
            raise RuntimeError(f"Login failed ({res.status_code}): {res.text}")
        
        logger.info("Authentication successful")
        token = res.text.replace('"', "")
        logger.debug(f"Token obtained (length: {len(token)})")
        return token
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during authentication: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {e}", exc_info=True)
        raise

def api_get_jobs(
    *,
    page: int = 1,
    page_size: Optional[int] = None,
    keywords: Optional[Iterable[str]] = None,
    keywords_logic: Optional[str] = None,              # e.g. "and" or "or"
    ids: Optional[Iterable[int]] = None,
    skill_ids: Optional[Iterable[str]] = None,         # ESCO skill URIs
    skill_ids_logic: Optional[str] = None,             # e.g. "and" or "or"
    occupation_ids: Optional[Iterable[str]] = None,    # ESCO occupation URIs
    occupation_ids_logic: Optional[str] = None,        # e.g. "and" or "or"
    organization_ids: Optional[Iterable[int]] = None,
    min_upload_date: Optional[str] = None,             # "YYYY-MM-DD"
    max_upload_date: Optional[str] = None,             # "YYYY-MM-DD"
    location_code: Optional[Iterable[str]] = None,     # e.g. ["GR","DE","FR"]
    sources: Optional[Iterable[str]] = None,           # e.g. ["linkedin"]
    token: Optional[str] = None,                       # pass in if you already have it
) -> dict:
    """
    Call POST /api/jobs with application/x-www-form-urlencoded body.

    Returns:
        dict: {'items': [...], 'count': N}
    Raises:
        requests.HTTPError on non-200 responses.
    """
    try:
        logger.info("=" * 60)
        logger.info("API GET JOBS REQUEST")
        logger.info(f"Page: {page}, Page size: {page_size}")
        logger.debug(f"Keywords: {list(keywords) if keywords else None}")
        logger.debug(f"Skill IDs: {list(skill_ids) if skill_ids else None}")
        logger.debug(f"Occupation IDs: {list(occupation_ids) if occupation_ids else None}")
        logger.debug(f"Location codes: {list(location_code) if location_code else None}")
        logger.debug(f"Sources: {list(sources) if sources else None}")
        logger.debug(f"Date range: {min_upload_date} to {max_upload_date}")
        
        if token is None:
            logger.info("No token provided, obtaining new token")
            token = get_token()
        else:
            logger.debug("Using provided token")

        # Query params
        params = {"page": page}
        if page_size is not None:
            params["page_size"] = page_size

        # Form body as list of (key, value) tuples to repeat array keys
        form: List[tuple[str, Union[str, int]]] = []

        def add_list(field: str, values: Optional[Iterable[Union[str, int]]]):
            if values is None:
                return
            for v in values:
                form.append((field, str(v)))

        add_list("keywords", keywords)
        if keywords_logic:
            form.append(("keywords_logic", keywords_logic))

        add_list("ids", ids)

        add_list("skill_ids", skill_ids)
        if skill_ids_logic:
            form.append(("skill_ids_logic", skill_ids_logic))

        add_list("occupation_ids", occupation_ids)
        if occupation_ids_logic:
            form.append(("occupation_ids_logic", occupation_ids_logic))

        add_list("organization_ids", organization_ids)

        if min_upload_date:
            form.append(("min_upload_date", min_upload_date))
        if max_upload_date:
            form.append(("max_upload_date", max_upload_date))

        add_list("location_code", location_code)
        add_list("sources", sources)

        headers = {"Authorization": f"Bearer {token}"}
        
        logger.info(f"Making POST request to {API}/jobs")
        logger.debug(f"Query params: {params}")
        logger.debug(f"Form data fields: {len(form)} items")
        
        resp = requests.post(f"{API}/jobs", params=params, headers=headers, data=form)
        
        logger.info(f"Response status: {resp.status_code}")
        
        if resp.status_code != 200:
            logger.error(f"API request failed with status {resp.status_code}: {resp.text}")
            raise requests.HTTPError(f"/jobs failed ({resp.status_code}): {resp.text}")

        result = resp.json()
        jobs_count = result.get('count', 0)
        items_count = len(result.get('items', []))
        
        logger.info(f"API returned {jobs_count} total jobs, {items_count} items in this page")
        logger.info("=" * 60)
        
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during API request: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in api_get_jobs: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    """
    Test the external queries API functions.
    """
    try:
        logger.info("Starting external queries test")
        
        # Minimal example
        # data = api_get_jobs(page=1, page_size=50)

        # With filters example
        # data = api_get_jobs(
        #     page=1,
        #     page_size=25,
        #     keywords=["data scientist", "python"],
        #     keywords_logic="and",
        #     skill_ids=["http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d"],
        #     skill_ids_logic="or",
        #     location_code=["GR", "DE"],
        #     sources=["linkedin"],
        #     min_upload_date="2025-08-01",
        #     max_upload_date="2025-09-16",
        # )

        # Current test
        logger.info("Testing api_get_jobs with Python and Java keywords")
        result = api_get_jobs(
            page=1,
            page_size=100,
            keywords=["python", "java"],
            keywords_logic="and",
            skill_ids_logic="or",
            occupation_ids_logic="or",
        )

        logger.info(f"Test completed successfully")
        logger.info(f"Result summary: {result.get('count', 0)} jobs found")
        logger.debug(f"Full result: {result}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)