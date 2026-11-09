# Android Release Signing

## v1.0.0 to v1.1.0

The original v1.0.0 APK was verified as package
`com.example.food_calorie_estimator`, version `1.0.0` / code `1`, and signed with
the local Android debug certificate. Its published SHA-256 matched the downloaded
APK. Starting with v1.1.0 (`1.1.0+2`), the project uses a dedicated production
release keystore configured through the ignored file `mobile/android/key.properties`.

Because the signing identity changes, Android will not install v1.1.0 as an in-place
update over v1.0.0. Users must remove the old app before installing the new one.
Uninstalling removes app-private data, including the local SQLite diary and saved
meal photos; arrange a data backup before reinstalling. The app does not provide an
automatic migration between these signing identities.

## Locally verified v1.1.0 APK

The release candidate at
`mobile/build/app/outputs/flutter-apk/app-release.apk` is a local build output and
is ignored by Git. It has not been uploaded as a public release. Local verification
on 2026-11-09 found:

- Size: 103,563,167 bytes.
- SHA-256: `b71e0eb55b2fa5e4bec0ce1ae7c3c237809f9540a97d603f84e5d07ca535c696`.
- Package: `com.example.food_calorie_estimator`.
- Version name/code: `1.1.0` / `2`.
- Minimum/target SDK: 24 / 36.
- `apksigner verify --verbose --print-certs`: valid APK Signature Scheme v2
  signature; no v1 signature was present.
- Signer certificate SHA-256: `c9e973cf8899bd79bfc31d4c6021ba53a9d503841e0aa948eb332ed8fbe922c7`.

The fingerprint matches the expected release fingerprint supplied for this build.
Verification inspected the APK signature; it did not read or disclose signing
credentials or private keystore contents. The user confirmed the final v1.1.0 APK
passed functional testing on a Samsung Android phone. This is user-reported
verification and was not independently reproduced as part of the repository checks.
The v1.0.0 public APK URL and checksum remain recorded in the root README.

The v1.1.0 release build must use the dedicated signing configuration. Gradle does
not fall back to debug signing. `key.properties` and keystore files are excluded by
`mobile/android/.gitignore`; keep both outside version control and secure the
keystore with an encrypted, independent backup.

## Create and store the release key manually

Run this locally in a terminal with a JDK `keytool` available. The tool prompts for
the keystore and key passwords interactively; do not add passwords to command-line
arguments or shell history.

```bash
umask 077
mkdir -p "$HOME/.android/food-calorie-estimator"
keytool -genkeypair -v \
  -keystore "$HOME/.android/food-calorie-estimator/release-v1.jks" \
  -storetype JKS \
  -keyalg RSA \
  -keysize 2048 \
  -validity 10000 \
  -alias food-calorie-estimator-v1
chmod 600 "$HOME/.android/food-calorie-estimator/release-v1.jks"
```

Create `mobile/android/key.properties` locally with restrictive permissions, then
fill in the values in a local editor without sharing them:

```bash
umask 077
touch mobile/android/key.properties
chmod 600 mobile/android/key.properties
${EDITOR:-vi} mobile/android/key.properties
```

The file format is:

```properties
storeFile=/absolute/path/to/release-v1.jks
storePassword=<enter locally>
keyAlias=food-calorie-estimator-v1
keyPassword=<enter locally>
```

Replace the password placeholders locally. Keep an encrypted backup of the
keystore and its credentials in separate secure storage, and verify that the backup
can be restored. Losing the key prevents future updates signed with that identity.

## Release checks

Before release, verify the package ID remains
`com.example.food_calorie_estimator`, version name/code are `1.1.0` / `2`, and the
release APK is signed by the dedicated certificate. Do not publish or install a
release artifact until the keystore is available, the signing configuration passes,
and the local-data migration impact has been communicated.

The local Android candidate has passed package/version/SDK and APK signature checks,
and final functional testing on a Samsung Android phone was confirmed by the user.
Separate release publication remains pending until the GitHub Release is created.
