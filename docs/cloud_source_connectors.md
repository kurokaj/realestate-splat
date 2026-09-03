# Cloud Source Connectors

This document records the setup and intended workflow for importing capture
media from Google Photos and iCloud into the local Buildvision3D tool.

## Current Status

- Local file upload is the currently supported source.
- Google Photos is the planned first cloud connector.
- iCloud Photos is not implemented. Apple does not provide a general-purpose
  server-side OAuth picker for browsing a user's iCloud Photos library in the
  same way as Google Photos Picker.
- Cloud media should be treated as a source reference until preprocessing.
  The selected video does not need to be copied in full to R2; preprocessing
  can download it temporarily, select frames, and upload only the selected
  frames and summaries to R2.

## Google Photos

### Account arrangement

The Google Cloud project and the Google Photos account do not need to be the
same account.

- Development account: owns the Google Cloud project and OAuth client.
- Main account: owns the photos and is added as an OAuth test user.

Google Photos requires user OAuth and does not support service accounts.

### Create the Google Cloud project

1. Sign in to the Google Cloud Console with the development account.
2. Create a project for the local Buildvision3D connector.
3. Select the new project.
4. Open **APIs and Services -> Library**.
5. Enable the Google Photos Picker API. If the Console presents the Photos
   Library API separately, do not enable broader access unless the application
   actually needs it.

The intended permission is the narrow Picker read-only scope:

```text
https://www.googleapis.com/auth/photospicker.mediaitems.readonly
```

### Configure the consent screen

In **Google Auth Platform** (the Console may show this as the OAuth consent
screen setup):

1. Open **Branding**.
2. Set the app name to `Buildvision3D`.
3. Set the user support email to the development account.
4. Set the developer contact email to the development account.
5. Save the Branding section.
6. Open **Audience**.
7. Choose **External**.
8. Add the main Google Photos account as a test user.
9. Open **Data Access** and add the Picker scope above.

For local testing, a public website or verified domain is not required. Leave
App domain, homepage, privacy policy, and terms URLs empty when the Console
allows them to be optional. Do not add `localhost` as an Authorized domain.

The expected local OAuth values belong in the client configuration instead:

```text
Authorized JavaScript origin:
http://localhost:8000

Authorized redirect URI:
http://localhost:8000/auth/google/callback
```

If Google requires a public homepage or privacy-policy URL before saving the
application, stop and review that validation message. Do not invent a public
domain for this local-only tool. A real owned and verified domain becomes
relevant when the application is hosted publicly or submitted for verification.

### Create the OAuth client

1. Open **APIs and Services -> Credentials**.
2. Select **Create Credentials -> OAuth client ID**.
3. Choose **Web application**.
4. Add the authorized origin and redirect URI shown above.
5. Create the client.
6. Download the client JSON file.

Keep the downloaded JSON outside Git. It contains a client secret. The
localhost URL inside the JSON, if present, is configuration rather than a
password: it tells Google where to return the authorization response.

### Store the local secret

Recommended local layout:

```text
secrets/google_photos_client.json
```

The `secrets/` directory and client JSON must be ignored by Git. Configure the
local application with the equivalent of:

```text
GOOGLE_PHOTOS_CLIENT_SECRETS=/app/secrets/google_photos_client.json
GOOGLE_PHOTOS_REDIRECT_URI=http://localhost:8000/auth/google/callback
```

When running the API directly outside Compose, use the corresponding absolute
host path instead of `/app/...`.

Do not put the client secret, OAuth refresh token, or access token in:

- source files
- `README.md` or documentation
- Postgres event rows
- browser-visible HTML or JavaScript
- R2 manifests

The backend should own the OAuth exchange and store tokens in a local secret
store or protected local file. The browser should receive only the picker
session information needed for the current selection.

### Intended local picker flow

1. User opens the Google Photos source selector in the local UI.
2. The API starts the OAuth flow if no valid local token exists.
3. Google redirects back to the exact localhost callback.
4. The API creates a Google Photos Picker session.
5. The UI opens the picker URI in the user's browser.
6. User selects one video or a set of images.
7. The API polls the picker session until selection is complete.
8. The API lists the selected media and displays filenames, type, and basic
   metadata for tagging.
9. User assigns the Buildvision3D role and location to the batch.
10. The manifest records the Google Photos item references and source metadata,
    rather than pretending the files originated from a local path.
11. Preprocessing downloads the selected source temporarily, applies the
    location settings, and uploads selected frames, hero images, summaries,
    and manifests to R2.
12. Temporary downloaded source media is deleted after the stage completes or
    fails.

### Localhost security

`localhost` means the callback is intended for the same computer running the
local API. Keep the API bound to `127.0.0.1` for local-only use and avoid
exposing port `8000` through a router or public tunnel.

The implementation must also:

- validate OAuth `state` on callback
- use the exact registered redirect URI
- request only the Picker read-only scope
- avoid logging authorization codes or tokens
- use short-lived picker media URLs immediately, since they expire
- keep the OAuth client JSON out of Git and backups that are not protected

An unverified-app warning is expected while the OAuth application is in
testing mode. Public launch and broader user access require Google's review
and verification process.

## iCloud

### Important platform limitation

iCloud Photos is not currently a drop-in equivalent to Google Photos Picker.
There is no supported general-purpose web OAuth flow that lets this local
FastAPI application browse a user's private iCloud Photos library and obtain
media in the same way.

The following approaches should not be used:

- scraping iCloud.com
- automating an iCloud web login
- storing an Apple ID password in the application
- treating an iCloud Photos web session as a stable API

### Viable future approaches

#### Native macOS or iOS helper

A small signed native helper can request the user's Photos-library permission
through Apple's Photos framework, show the system media picker, and export the
user-selected video or images to a local handoff directory. Buildvision3D then
uses the existing local upload path.

The helper should return:

- selected local file paths or a temporary archive
- original filename and media type
- capture timestamp where available
- image/video dimensions and duration where available
- an import batch identifier

The existing upload code can then assign one role and one location to the
whole batch and create or append the raw manifest.

#### iCloud Drive route

If the user explicitly exports or saves the media into iCloud Drive, a future
connector could use a native macOS file-provider flow to make those files
available locally. This is an iCloud Drive file workflow, not direct browsing
of the iCloud Photos library.

### Planned iCloud UX

The UI can expose an iCloud option now as a disabled or clearly marked
placeholder. When implemented, it should follow the same contract as Google
Photos:

1. Select the source provider.
2. Select a batch of images or one coverage video.
3. Assign the batch role and location once.
4. Reject a second coverage video for the same location unless the user
   explicitly confirms replacement.
5. Store source references and metadata in the raw manifest.
6. Let preprocessing create the R2 outputs.

## References

- [Google Photos API app configuration](https://developers.google.com/photos/overview/configure-your-app)
- [Google Photos authorization](https://developers.google.com/photos/overview/authorization)
- [Google Photos Picker sessions](https://developers.google.com/photos/picker/guides/sessions)
