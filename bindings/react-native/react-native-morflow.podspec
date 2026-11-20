# The iOS side of react-native-morflow: the C++ module, and Morflow's native
# libraries as frameworks: morflow_capi and one morflow_pack_<pack> per pack
# the app's package.json lists under morflow.packs. They are prepared during
# `pod install`, so run it again after changing the packs.
require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

# The app's root directory, where its package.json is: the parent of ios/.
app_root = ENV["MORFLOW_APP_ROOT"] || File.dirname(Pod::Config.instance.installation_root.to_s)
frameworks = File.join(__dir__, "ios", "Frameworks")
unless system("node", File.join(__dir__, "scripts", "fetch-native.js"), "ios", "--app", app_root, "--out", frameworks)
  raise "react-native-morflow could not prepare Morflow's native libraries; see the message above."
end

Pod::Spec.new do |s|
  s.name         = "react-native-morflow"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]
  s.platforms    = { :ios => min_ios_version_supported }
  s.source       = { :git => "https://github.com/JiraPit/Morflow.git", :tag => "react-native-v#{s.version}" }

  s.source_files = "ios/**/*.{h,m,mm}", "cpp/**/*.{h,cpp}"
  s.vendored_frameworks = Dir.chdir(__dir__) { Dir.glob("ios/Frameworks/*.xcframework") }

  install_modules_dependencies(s)
end
