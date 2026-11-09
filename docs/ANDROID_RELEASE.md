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
meal photos; arrange an independent data backup before reinstalling. The app does
not provide an in-app export or automatic migration between these signing
identities. If no independent backup is available, uninstalling may permanently
lose local data.

## Published v1.1.0 APK

The published release APK at
`mobile/build/app/outputs/flutter-apk/app-release.apk` was locally verified on
2026-11-09 before publication. Verification found:

- Size: 103,563,167 bytes.
- SHA-256: `b71e0eb55b2fa5e4bec0ce1ae7c3c237809f9540a97d603f84e5d07ca535c696`.
- Package: `com.example.food_calorie_estimator`.
- Version name/code: `1.1.0` / `2`.
- Minimum/target SDK: 24 / 36.
- `apksigner verify --verbose --print-certs`: valid APK Signature Scheme v2
  signature; no v1 signature was present.
- Signer certificate SHA-256: `c9e973cf8899bd79bfc31d4c6021ba53a9d503841e0aa948eb332ed8fbe922c7`.

The signer fingerprint is public certificate metadata. APK verification does not
reveal the private keystore key or its passwords. Final functional testing passed
on a physical Samsung Android phone. The v1.0.0 public APK URL and checksum remain
recorded in the root README.

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

Before the v1.1.1 release, verify the package ID remains
`com.example.food_calorie_estimator`, version name/code are `1.1.1` / `3`, and the
APK is signed by the same dedicated certificate as v1.1.0. Do not publish or
install a release artifact until the signing configuration passes and the local-data
retention impact has been communicated. Direct update from v1.0.0 remains
incompatible; its debug signing identity differs from the dedicated release signer.

The isolated v1.1.1 candidate APK was built and checked locally. It is not a
published asset and remains subject to physical-device testing:

- Package: `com.example.food_calorie_estimator`.
- Version name/code: `1.1.1` / `3`; minimum/target SDK: 24 / 36.
- Size: 103,564,315 bytes; SHA-256:
  `09de5811df6db37647f6367d0f8edd02b971b628ecb53a43e5622e5527dffd77`.
- `apksigner` verified APK Signature Scheme v2. The certificate SHA-256 matches
  the dedicated v1.1.0 release signer recorded above.
- The effective release manifest has `allowBackup=false` and references both
  backup-rule resources. The packaged legacy, cloud, and device-transfer rules
  exclude app-private data domains.

## Local diary backup policy

The v1.1.1 candidate sets `android:allowBackup="false"` and references explicit
legacy full-backup and Android 12+ data-extraction rules. The rules exclude app
data from cloud backup and device transfer, including the SQLite diary and saved
meal photos. Android documentation notes that device-to-device transfer behavior
may vary by manufacturer, so verify behavior on supported devices. See the official
[Android Auto Backup documentation](https://developer.android.com/identity/data/autobackup).
This is not a data export feature. App-private data may be lost on uninstall or
device change; users should keep an independent copy of data they need. The policy
does not alter the database schema, migration, or photo-storage format.
These rules cover Android cloud backup and Android device transfer. Android 16
QPR2 cross-platform transfer to iOS uses a separate configuration section; no
cross-platform diary migration is implemented or claimed.

The published v1.1.0 manifest did not define backup exclusions, so Android's
default backup behavior may have included app data when backup was enabled on a
device. The v1.1.1 rules apply to subsequent backup and transfer operations; they
do not delete copies already held by an Android backup provider.

## Flutter build-path metadata

Inspection of the published APK and the isolated v1.1.1 candidate found an absolute
build-root string in each ABI's Flutter `libapp.so`. In the published artifact it
identifies the developer home directory; the isolated candidate contains its
temporary build root instead. The string reveals build-environment path metadata,
not a credential or the source files themselves. A standard release build retained
this metadata. A diagnostic build with Flutter's supported obfuscation and
split-symbol options retained the same build-root reference and emitted an ELF
DWARF warning, so those options were not adopted for this candidate. No unsupported
binary rewriting or unverified workaround was applied. Removing the metadata would
require a verified approach that preserves usable symbolized diagnostics.
