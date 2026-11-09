import java.util.Properties
import org.gradle.api.GradleException

plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

val releaseSigningPropertiesFile = rootProject.file("key.properties")
val releaseSigningProperties = Properties().apply {
    if (releaseSigningPropertiesFile.isFile) {
        releaseSigningPropertiesFile.inputStream().use { load(it) }
    }
}

val releaseSigningConfig = android.signingConfigs.create("release") {
    releaseSigningProperties.getProperty("storeFile")
        ?.takeIf(String::isNotBlank)
        ?.let { storeFile = rootProject.file(it) }
    storePassword = releaseSigningProperties.getProperty("storePassword")
    keyAlias = releaseSigningProperties.getProperty("keyAlias")
    keyPassword = releaseSigningProperties.getProperty("keyPassword")
}

gradle.taskGraph.whenReady {
    val releaseVariantRequested = allTasks.any { task ->
        task.project == project && task.name.contains("release", ignoreCase = true)
    }
    if (releaseVariantRequested) {
        val missingProperties = listOf(
            "storeFile",
            "storePassword",
            "keyAlias",
            "keyPassword",
        ).filter { releaseSigningProperties.getProperty(it).isNullOrBlank() }
        val keystorePath = releaseSigningProperties.getProperty("storeFile")
        val keystoreFile = keystorePath
            ?.takeIf(String::isNotBlank)
            ?.let { rootProject.file(it) }
        val invalidKeystore = keystoreFile?.let { !it.isFile || !it.canRead() } ?: true

        if (missingProperties.isNotEmpty() || invalidKeystore) {
            val missingDescription = if (missingProperties.isEmpty()) {
                "none"
            } else {
                missingProperties.joinToString()
            }
            val keystoreDescription = if (invalidKeystore) {
                "storeFile must point to a readable keystore"
            } else {
                "keystore path is valid"
            }
            throw GradleException(
                "Release signing is not configured. Add the required values to " +
                    "mobile/android/key.properties (missing: $missingDescription; " +
                    "$keystoreDescription). Release builds never fall back to " +
                    "debug signing."
            )
        }
    }
}

android {
    namespace = "com.example.food_calorie_estimator"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.food_calorie_estimator"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        // Uses the version code from pubspec.yaml. When using split APKs, 1000 * ABI_VERSION
        // is added automatically by Flutter. (https://developer.android.com/studio/build/configure-apk-splits#configure-APK-versions)
        // You can force using the value of versionCode by specifying the `-P force-version-code-ignoring-abi=true`
        // flag during build.
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            signingConfig = releaseSigningConfig
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
