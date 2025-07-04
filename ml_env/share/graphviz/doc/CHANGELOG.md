# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.50.0] – 2021-12-04

### Added

- hard-coded lookup tables for fallback font metrics for more fonts and font
  variants
- a new `gvputs_nonascii` API function has been implemented for GVC I/O with C
  escaping

### Changed

- Check for existence of `dl_iterate_phdr(3)` and if it is available, prefer
  using it instead of iterating over `/proc/self/maps` for computing `libdir`.
- A limit on GVC config files of 100000 bytes has been removed.
- MD5 checksums of release artifacts are no longer provided. SHA256 checksums
  are still provided and these should be used instead.
- when cross-compiling, the `dot -c` is no longer run during installation
- `$CMAKE_INCLUDE_PATH` is no longer manually configured in the CMake build
  system

### Fixed

- remove Bashism from `gvmap.sh` #2151
- Lefty artifacts are no longer installed when Lefty is disabled #2153
- Smyrna artifacts are no longer installed when Smyrna is disabled
- calling convention mismatches in delaunay.c’s GTS code
- impure assertion in `jacobi`
- undefined behavior in libgvc’s reading of little endian numbers
- boldness of `agnxtsubg` in cgraph man page
- parameter `name` in `gvusershape_find` prototype corrected to a const pointer,
  to match the implementation
- xdot JSON output is not valid JSON #1958
- fix uninitialized read of `pid` in `_sfpopen` on Windows
- claimed minimum CMake version supported has been corrected to 3.9

## [2.49.3] – 2021-10-22

### Fixed

- gvpr "split", "tokens", and "index" functions produce incorrect results #2138.
  This was a regression that occurred between 2.47.1 and 2.47.2.

## [2.49.2] – 2021-10-16

### Changed

- Lefty is disabled by default in the Autotools build system. To re-enable it,
  pass `--enable-lefty` to `./configure`. In a future release, Lefty will be
  removed.
- remove PHP5 support in SWIG bindings

### Fixed

- Msys experimental packages are included in release artifacts #2130
- CMake build system incorrectly aliases gv2gml to gml2gv #2131
- Gv2gml Doesn't escape quotes in attributes #1276
- GVPR incorrectly understands color schemes #1956

## [2.49.1] – 2021-09-22

### Changed

- the CMake build system installs gzipped man pages if `gzip` is available #1883
- CMake projects using Graphviz as a subproject (`add_subdirectory`) can now
  link against and use `gvc`.

### Fixed

- various problems in the generation of Javascript bindings
- 2.48.0: test suite is failing #2112
- Ensure correct file-level dependency for generated file in cmake generated
  projects #2119
- compile failures with a C++20-compatible toolchain #2122
- compile errors on macOS when using Bison 3.8 #2127
- Make Graphviz buildable as a cmake subproject/subdirectory #1477
- Header not found in Cmake project #2109

## [2.49.0] – 2021-08-28

### Added
- a very basic C++ API for a subset of the functions in lib/cgraph and
  lib/gvc, allowing a graph to be rendered from DOT source to a
  specified format. The new API is available through two new
  libraries: lib/cgraph++ and lib/gvc++. It is experimental, meaning
  that it might have breaking changes also in upcoming patch or minor
  releases (towards #2001)
- CMake builds now support an `with_expat` option that allows the support for
  using HTML-like labels through the optional expat library to be explicitly
  enabled (default) or disabled
- CMake builds now support an with_zlib option that allows the support for
  raster image compression through the optional zlib library to be explicitly
  enabled (default) or disabled

### Changed

- the CMake build system now enables `-Wextra` when building C++
- some Cgraph functions that take `char*` arguments that they do not modify have
  been updated to take `const char*` arguments #634
- incorrectly using the `layout` attribute on anything other than a graph now
  results in a warning about this being invalid #2078
- `edgepaint` accepts more standard `--` prefixed command line arguments and
  rejects invalid options #1971
- improved detection of Lefty dependencies in the Autotools build system
- libexpr rejects printing the time (`%t`) if no format is provided
- `-DDATE=…` option in the CMake build system has been removed
- the Autotools build system no longer writes the DATE file and the portable
  source tarball no longer includes this

### Fixed

- The attached dot file causes a segfault when processed #2095
- fix typos and update URLs in `edgepaint` usage text and man page
- Fix clang's undefined behavior warning in dotLayout
- gvpr doesn't build on macOS but MKDEFS_EXECUTABLE points to wrong
  directory #2101
- the generated gdefs.h header is no longer installed
- `ccomps` out-of-memory message no longer incorrectly refers to `gc`
- do not abort when `calloc(0, x)` or `calloc(x, 0)` in `gcalloc` return `NULL`
- failed Exshort_t type discrimination #1799
- dot manpage is in wrong directory on Windows #1936
- CMake builds fail when when the ltdl library is not available even if the
  `enable_ltdl` option is `ON`
- CMake builds fail when when the optional `zlib` library is not available
- fix graph rotation in quartz plugin

## [2.48.0] - 2021-07-17

### Added
- a new C++ test infrastructure based on ctest and Catch2 towards #2002
- support for test coverage analysis with
  [lcov](http://ltp.sourceforge.net/coverage/lcov.php) and
  [test coverage visualization in GitLab MRs](https://docs.gitlab.com/ee/user/project/merge_requests/test_coverage_visualization.html)

### Changed

- updated Graphviz bug report URL in the Autotools build system
- Fix `WIN32` path of `gvc.def` specified in `libgvc_la_LDFLAGS`
- the CMake build system now not only checks for Bison, but also ensures the
  found version is recent enough #1916

### Fixed

- ortho's eqEndSeg incorrectly modifies its arguments #2047
- Autotools enables -Wtrampolines and -Wlogical-op for Clang #2066
- node_distinct_coloring failure due to out-of-memory now reported correctly
  instead of referring to a failure to open lab_gamut
- Fix a typo `GD_LIBS` to `GDLIB_LIBS` in `tclpkg/tcldot/Makefile.am` !2022
- Autotools build system sets libgd variables now instead of incorrectly setting
  GTK variables
- HTML strings used as labels are distinguishable in GXL output by
  `kind="HTML-like string"`
- a Bashism removed from the Autotools build system
- when Criterion is available, the `command_line` test binary is no longer built
  and installed by default, but rather during `make check`
- round-tripping a file through ``gv2gxl`` and then ``gxl2gv`` no longer causes
  HTML-like labels to become non-HTML like labels #517
- fix ODR violation by including the ortho object files in the gvc lib also for
  CMake and MSbuild #2096

## [2.47.3] - 2021-06-19

### Changed

- marginally more accurate computations in Smyrna sphere projection
- Perl is no longer required to build Graphviz #2067
- nop more reliably returns success and failure exit statuses
- implicit 63 character limit on plugin names is removed in GVC
- the packaging work flow on CentOS 7 now selects the Python 3 bindings, instead
  of Python 2 bindings
- remove Python 2 bindings #1992
- improved thread-safety in Graphviz bindings Makefile

### Fixed

- no longer pass libcommon to the linker twice in mm2gv when building with CMake
- Quartz plugin is now compiled with explicit `--tag=CC` to libtool #2065
- out of bounds read when attempting to load a plugin whose name is ≥63
  characters
- out of bounds read when examining a registered plugin whose name is ≥63
  characters
- do not `fclose(NULL)` in gvmap
- Assertion error when using `dot` in `ortho.c` in `convertSPtoRoute` in
  graphviz 2.47.2 #2082. This was a regression introduced in 2.47.2.

## [2.47.2] - 2021-05-26

### Added

- CMake option `use_sanitizers` which enables building with address and
  undefined behavior sanitizer

### Changed

- $PATH is no longer assumed to be "/bin:/usr/bin:/usr/local/bin" if unset
- test suite no longer assumes `python3` exists #2049
- CMake build system locates Python 3 before calling it
- diff and grep are no longer required to build Graphviz on Windows

### Fixed

- Uninitialized variable read in delaunay_tri
- potentially mismatched format string in tclpkg
- `gvToolTred` is now exported from gvc.dll on Windows mirroring the behavior on
  Unix platforms.

## [2.47.1] - 2021-04-17

### Changed

- More detailed error messages when opening input file fails

### Fixed

- Windows build thinks xdg-open can be used to open a web browser #1954
- lab_gamut_data misses a value #1974
- xdot man page does not document some functions #1957
- Superfluous empty `@param` in documentation #1977
- PIC renderer does not work and probably never has #131
- dot conversion to dia format #689
- memory leak of reference-counted HTML strings
- Align rank from bottom in dot graph #1339
- Fix for TBbalance attribute code #1980
- HTML parser error with single closing square bracket in table row #1893
- reference counted strings put the HTML bit in the middle of the reference
  count #1984
- &amp;amp; escape disappearing #797
- miscalculation of minimum rank on large graphs
- AddressSanitizer: strcpy-param-overlap in gvconfig_libdir when
  running dot -c #1994
- fix reuse of va_list in pov rendering

## [2.47.0] - 2021-03-15

### Changed
- The edges in JSON output are ordered now !1728
- remove regex usage #1919
- RxSpencer is no longer a dependency on Windows
- gvmap.sh is compatible with POSIX shells in addition to ksh
- sed is no longer a build dependency on Windows
- SHA256 checksum generation? #1955

### Fixed
- Fix gvpr -? to actually print usage and exit non-zero
- gvpr is not built by CMake #1878
- typos in gpcanvas.c #1927
- memory leak in libmingle
- private inheritance in IncVPSC #1874
- broken sorting in nearest_neighbor_graph_ann.cpp #1938
- memory leak in ANN bridge
- gvpr on Windows does not support absolute paths #1780
- buffer overflow in unflatten
- agxbputc macro does not bracket its arguments #1814

## [2.46.1] - 2021-02-13

### Added
- Support for building against Guile 2.2
- Portable source is now also offered as a .tar.xz

### Changed
- CentOS/RHEL 6 is no longer supported
- Vestiges of Qt4 support have been removed
- C++11 support is now required of the C++ compiler used to build Graphviz
- C99 support is now required of the C compiler used to build Graphviz
- Question about userout() function in agerror.c #1924
- The minimum version of Python required to run the test suite is 3.6

### Fixed
- memory leak in label construction
- gvedit compilation errors out, but works if manually compiled with qt5 #1862
- incorrect HTML BR attribute parsing code #1913
- broken overflow checks in RectArea #1906
- various memory leaks !1699
- Fix bad free in lefty !1709
- typo in pathcross #1926
- Out-of-bounds write caused by incorrect error handling of malloc in genUserdata #1928
- Offer .tar.xz files too #454
- Header file graphviz_version.h has no include guards #1929
- regression: newlines embedded in quoted labels / node names are not preserved in 2.46.0 #1931
- Properly fill graphviz_version.h !1706

## [2.46.0] - 2021-01-18

### Added
- Cgraph's agxbuf API gained a new function agxbdisown(), for dissociating
  backing memory from the managed buffer
- Build system support for the Elbrus 2000 CPU, thanks to Michael Shigorin

### Changed
- Cgraph's agheap() API has been removed
- Autotools build system support for eFence has been removed
- Building Graphviz with ICC defaults to -O2 instead of -O0
- Build system work arounds for GCC 3 have been removed
- Incomplete support for running the test suite under CMake has been removed
- Portable source tarballs now use the “ustar” POSIX format
- Minimum version of Flex required to build Graphviz is now 2.5.2
- Minimum version of Bison required to build Graphviz is now 3.0
- Minimum version of CMake required to build Graphviz using CMake is now 3.1

### Fixed
- gvpr: line numbers in gvpr errors/warnings are incorrect #1594
- URL typo in patchwork man page
- Escaped backslashes are not correctly handled when producing xdot with dot #165
- heap-over-flow(off-by-null) in lib/common/shapes.c #1700
- Windows MSBuild executables have the wrong version #1745
- Cast Overflow at pango_textlayout #1314
- x11 back end segfaults if display is unavailable #1776
- typo in cmd/gvpr/lib/clustg #1781
- Segfault in dot #1783
- Incorrect 'Arrow type "s" unknown' error #1444
- segfault on reading 0x10 #1724
- Null-dereference READ (144736912) #1676
- "Warning! PATH too long installer unable to modify PATH!" using CMake Windows installer and PATH length > 1024 #1770
- gvedit -? gives "option - unrecognized - ignored" instead of showing usage #1813
- lefty is not built for Windows (fixed for MSBuild builds only) #1818
- a failure to detect OpenGL glGenTextures() errors has been corrected
- sfio does compile time benchmarknig #1422
- iffe "lib" check always succeeds when compiler optimises #1521
- syntax error near text who is not present #1411
- Explicitly links with libstdc++; should allow libc++ if appropriate #163
- A macOS file that was erroneously excluded from portable source tarballs has
  been restored
- Add option -? for usage to diffimg
- Add option -? for usage to dotty
- Add option -? for usage to lneato
- Add option -? for usage to vimdot
- Fix smyrna -? to actually print usage instead of error
- Fix edgepaint -? to actually print usage instead of error
- Remove '"' from usage text in non-Windows version of dotty
- Correct misspelled 'smyrna' in usage
- Fix edgepaint -o option
- Correct shebang of gvmap.sh to use ksh
- Fix gvmap.sh -? option to exit with zero exit status
- Graphviz doesn't build on MacOS with the latest libc++ #1785
- make fails if ps2pdf is not installed (using autotools) #1763
- multiple graphs to file output causes a segfault #1845
- lefty PTY functionality relies on file descriptor implementation details #1823
- buffer overflow in fdpgen
- Crashes by VRML output when current directory is not writable #793
- Segmentation fault when newrank=true #1221
- sfdp craches #236
- fdp segmentation fault with GK=0 #1290
- fdp crash #1865
- Graphviz always crash with this simple dot file #167
- Seg fault in dot #1771
- gml2gv doesn't handle some attributes correctly #1869
- Add missing circo, fdp, neato, osage, patchwork, sfdp & twopi tools to Windows builds (copies of dot)
- Add gv2gml tool to CMake (copy of gml2gv on Windows, symlink to gml2gv otherwise)
- Regression: fdp generates internal names in the output #1876
- Regression: fdp assertion error on cluster in edge #1877
- Regression in id / &lt;title&gt; in svg for twopi #1907

## [2.44.1] - 2020-06-29

### Added
- applied RH patches (from graphviz-2.42.2-8.fc32.src.rpm)
  - graphviz-2.42.2-coverity-scan-fixes.patch
  - graphviz-2.42.2-dotty-menu-fix.patch
  - graphviz-2.42.2-ocaml-allow-const-cast.patch
- some allocation failures that could previously allow memory corruption now exit
- lab_gamut.3.pdf is no longer included in release archives

### Changed
- Windows binaries available at https://www2.graphviz.org/Packages/ instead of
  https://ci.appveyor.com/project/ellson/graphviz-pl238
- Retarget Windows builds to Visual Studio 2019 and toolset v142

### Fixed
- Released Ubuntu packages does not contain language bindings for Python3 #1737
- Neato's hier mode is broken since v2.44.0 #1726
- Segmentation fault (core dumped) #1436

## [2.44.0] - 2020-04-08

### Added
- New SGD mode in neato (thanks [Jonathan Zheng](https://gitlab.com/jxz12/graphviz/-/tree/sgd))
- Add pkg-config files !1322
- tred: add feature to output removed edges to stderr upon request !1326
- Fix issue #1671: Workaround: avoid creating a virtual edge loop. !1328
- Add riscv64 to host_cpu configure.ac !1329
- lib/cgraph: include empty malloc.h from subdir include !1332
- lib/gvpr: compile mkdefs with $(HOSTCC) rather than $(CC) !1333
- lib/vpsc: rename bcopy->b_copy !1334

### Fixed
- MSB4018 The NativeCodeAnalysis task failed unexpectedly. #1481

## [2.42.4] - 2020-04-05

### Added
- Include all test files in distro !1341
- host_cpu add mips64 platform !1325
- Correct description of 'port' syntax in manual !1324

### Fixed
- svg output displays TITLE of %3 if graph had no name #1376
- XML errors in generated SVG when URL attribute contains ampersand (&) #1687
- Test files missing from source distributions #1647
- SVG error for "g.transform.scale " in graphviz version 2.43 #1605

## [2.42.3] and earlier

```

October 9, 2019
    - Release 2.42.3
	- Merge 1316, 1317, 1319, 1320
	- Patches #1591, #1596
	- Add Fedora 32 builds
	- CI/CD fixes
	- Documentation (Warning about HTML label usage)

September 17, 2019
    - Release 2.42.2 - ( Never fully released due to CI/CD hardware issues ) 
    - Fix deployment issues.  Builds can now be found under:
	             http://www2.graphviz.org/Packages/
July 17, 2019
    - Release 2.42.1
    - Fix deployment issues.  Builds can now be found under:
	             http://www2.graphviz.org/Packages/
July 4, 2019
    - Release 2.42.0
    - Fixes quite a few bugs
September 22, 2017
    - Move master repo to GitLab: https://gitlab.com/graphviz/graphviz
December 21, 2016
	- Remove usage of ast_common.h
December 20, 2016
    - Release 2.40.0
        - network-simplex fixes and optimization (Stephen North)
	- built-in tred tool now available in the various swig generated
	language bindings (John Ellson)
	- number rounding added to SVG renderer (same as PS and TK rounding)
	to aid regression testing. (John Ellson)
	- additional regressson test framework, used in Travis CI builds. (Erwin Janssen)
	- PHP7 support (requires swig-3.0.11 or later). (John Ellson)
	- Allow user to specify clustering algorithm in gvmap. (Emden Gansner)
	- Add Sierpinski graph generator to gvgen. (Emden Gansner)
	- Extensive code cleanup (Erwin Janssen)
	- Removal of libgd source - use vanilla libgd from separate install 
	- Windows builds (Erwin Janssen)
	- Appveyor CI for automated Windows build testing (Erwin Janssen)
	- Travis CI for Fedora/Centos builds (Erwin Janssen)
	- Added JSON output format, -Tjson  (Emden Gansner)
	- New curved arrowhead, cylinder node shape.
	- Resolves bugs: 2599, 1172
June 18, 2016
	- Experimenting with Travis CI
February 13, 2016
	- Add cylinder shape for databases.
	- Free installed plugins
	- Update makefile for dot so that the using libpanco_C in the static build include PANGOFT2
        as well as PANGOCAIRO_LIBS (needed for some versions of Ubuntu)
February 1, 2016
	- Add json output format
April 26, 2015
	- output class value in svg files
September 9, 2014
	- Add plain shape for use with HTML-like labels.
August 12, 2014
	- Add icurve arrowhead.
July 28, 2014
	- Revert to old, translate to origin semantics in neato, etc. Add flag notranslate if that is
          what the user desires.
April 13, 2014
	- Release 2.38.0
	- Resolves bugs: 2409, 2413, 2417, 2420, 2422, 2423, 2425
March 27, 2014
	- Enable packing for dot
	- Allow scaling to work for all non-dot layouts
March 9, 2014
	- Add overline text characteristic.
March 4, 2014
	- Fix bugs in gvpr and gv.cpp so edges can be created in subgraphs.
	- Add edgepaint program for coloring edges to make them easier to tell apart.
	- Modify neato to avoid unnecessary translations of output. This allows positions
	given on input to remain the same on output.
	- Fix swig java package to work and support gv.renderresult.
	- Fix test for the absence of layout (old test relied on statically allocated Agraphinfo_t).
	- HTML-like tables and cells can now specify which borders should be drawn.
	- The fixedsize attribute now takes the value "shape" which allows labels much larger than the
	node shape.
January 11, 2014
	- Release 2.36.0
	- Resolves bugs: 2372, 2384, 2388, 2391, 2392, 2383, 2395, 2401, 2406
	- Various MacOS Fixes from Pixleglow.
	- Remove old libgraph sources from distributions.
	- Move master git repo to github.com
September 15, 2013
	- Add <S> element for strike-through to HTML-like labels.
September 6, 2013
	- Release 2.34.0
	- New version of xdot supporting inline text characteristics such as <b> and
	version-specific output based on xdotversion
	- Resolves bugs: 2325, 2326, 2333, 2334, 2337, 2338, 2340, 2343,
		2345, 2346, 2349, 2350, 2351, 2352, 2353, 2354, 2357, 2359
	- Resolves Redhat bug: BZ#847458
August 21, 2013
	- Added mingle command and library for edge bundling
August 1, 2013
	- Release 2.32.0
	- New version of xdot format, annotating gradient color schemes
	- Support for reading pdf images using poppler
	- Lefty/dotty/lneato now accept anonymous graphs
July 2, 2013
	- Add star node shape
	- Add two-tone (non-gradient) fill
February 14, 2013
	- Release 2.30.1
	- various build fixes
January 13, 2013
	- Release 2.30.0
	- Replaced libgraph with libcgraph; use of libgraph is now deprecated
	- New ranking algorithm that allows multiple subgraph constraints
November 27, 2012
	- Add graphml2gv and gv2gml to Windows package.
September 25, 2012
	- Support edges using curved arcs.
August 16, 2012
	- Added new shapes used by the synthetic biology community.
July 12, 2012
	- For HTML-like labels, provide rounded cells, and dashed or dotted borders.
	- Add lcurve and rcurve arrowheads.
	- Add prototype pie chart and striped fills.
	- Support insets in treemaps to make containment clear
June 7, 2012
	- Add random rooted tree generation to gvgen
February 29, 2012
	- Allow GVPRPATH to specify that the default path be prepended or appended to it.
February 27, 2012
	- Support arbitrary lists of layers; allow the user to specify arbitrary layers for output.
February 24, 2012
	- A collection of gvpr scripts, which were part of the source package, are now
	installed in <prefix>/share/graphviz/gvpr, and the that path is used as part of th
	default built-in path for gvpr.
February 15, 2012
	- Update libexpr to reflect Glenn Fowler's changes including scopes for variables.
February 9, 2012
	- Add next graph variable to gvpr
February 8, 2012
	- Modify dot and fdp so that a cluster's margin attribute will affect the space
	  between the bounding box and nodes
January 26, 2012
	- Modify the dijkstra tool to use only directed edges
	- Output numbers without quotes if quotes are not needed on input
	- Support gradient fill
January 23, 2012
	- Provide support for webp images
January 17, 2012
	- Fix tapered edges to use the dir attribute and arrowhead
September 21, 2011
	- Add imagepath attribute
	- Add help functionality to Graphviz.app
August 24, 2011
	- Add <B>,<I>,<U> to html strings via svg
August 16, 2011
	- Add tapered edges
August 3, 2011
	- Add support for external labels
July 14, 2011
	- Add initial implementation of graphml2gv
July 8, 2011
	- Add basic horizontal and vertical rules to html tables
May 6, 2011
	- Release 2.28.0
	- incremented library api version:
	    libcdt, libgraph, libcgraph, libgvpr, libgvc
	- Add gvmap, cluster and gvmap.sh
	- Deprecate dotty; prefer gvedit
	- Add patchwork supporting squarified tree maps
	- Add ordering as a node attribute
	- Fix problems with font resolution
	- Fix problems with text placement
	- Fix twopi to set root attribute
	- Make available layouts and formats available via the API
	- Fix error message system so that an application can capture the messages
	- New Qt-based version of gvedit
	- New attributes and features for sfdp
	- gvgen now allows the user to specify graph name and node name template
	- Make overlap=false denote overlap=prism
	- More efficient xdot library
	- HTML-like labels provide ID
	- Fixed bugs: 1480 1980 2044 2087 2088 2089 2091 2093 2094 
	2095 2101 2102 2103 2104 2112 2113 2118 2128 2129 2139 2149 
	2157 2113 2159 2160 2161 2163
March 31, 2011
	- Add many new gvpr scripts to release package
	- Add scale attribute to twopi
October 14, 2010
	- Add <B>,<I>,<U> to html strings via cairo
February 15, 2010
	- migrated to 2005 version of cdt 
January 26, 2010
	- Release 2.26.3
	- libcgraph.so   version bumped from 4 to 5 due to API changes
	- Allow ranksep to specify multiple radial differences in twopi
	- Allow the user to specify all pairwise distances in neato with
	- Fixed bugs: 1280 1409 1567 1583 1624 1631 1655 1708 1709
	1727 1784 1792 1798 1800 1813 1814 1830 1831 1833 1836 1839
model=mds
December 10, 2009
	- Release 2.26.0
	- Core features:
		- added: "smyrna" - a new opengl-based viewer for large graphs
		- added: rudimentary "gml2gv", "gv2gml" converters
		- extended support for various image formats in node images
		- removed vestiges of codegens, now all putput formats supported
		  through plugins.  Dropped some output formats for which
		  plugins have not been developed: -Tdia, -Tmif
		- gvpr converted to a library; additional array handling and
		  text processing functions added; language extended to allow 
		  multiple BEG_G/N/E blocks. 
		- allow background images specified via xdot
	- Ports added/dropped from nightly builds:
	  (The dropped ports could probably be re-added if there was demand.)
		- added MacOSX SnowLeopard  (multiarch:  i386/x86_64/ppc)
		- added Fedora 12 (i386, x86_64)
		- added Fedora 13 (Rawhide) (i386, x86_64)
		- dropped Fedora 7 (i386, x86_64)
		- dropped Fedora 8 (i386, x86_64)
		- dropped RHEL 3 (i386, x86_64, ia64)
		- dropped Ubuntu 8 (i386)
	- Fixed bugs: 1683 1713 1718 1720 1738 1747 1759 1770 1776 1786
	  1799 1816 1827

June 16, 2009
	- Release 2.24.0
	- Core:
		- add new layout engine for large graphs: sfdp
		- add new layout engine for nested graphs: osage
        - pack library extended to handle array-based packing modes
          using array bounds, aspect ratio, user-controlled sorting, etc. 
	- Fixed bugs: 1515 1590 1598 1601 1605 1607 1609 1610 1611 1614
	1615 1617 1625 1628 1634 1635 1640 1641 1642 1646 1649 1651 1652
	
March 13, 2009
	- Release 2.22.2
		- fix for buffer overflow (present in 2.22.0 and 2.22.1)
	- Fixed bugs:
		1602

March 9, 2009
	- Release 2.22.1
		- build fixes for Visual Studio and for FreeBSD
March 3, 2009
	- Release 2.22.0
	- Core:
		- libgvc api changed, version bumped.  Affects third party
		  applications using libgvc.
		- plugin api changed, version bumped.  Affects third party
		  plugins for graphviz.
		- 90% conversion to cgraph has been done, but not enabled yet,
		  (and yes, its true what they say about the last 10% )
		- drop libagraph from distribution  (use libcgraph)
		- layout code completely converted to floating point.
		- new "dot -P" option for generating a graph of available
		  plugins.
		- registered MIME type:  text/vnd.graphviz for .gv files
		- rename files from .dot to .gv to avoid conflict with
		  Word templates.  .dot still supported, but deprecated.
		- new command: mm2gv   (matrix-market graph file conversion)
		- rename commands:	dot2gxl -> gv2gxl
					gxl2dot -> gxl2gv
	- Plugins:
		- new rsvg plugin for support of node shapes in SVG format
		- new gs plugin for support of node shapes in PS format
		- new lasi plugin for support of UTF-8 characters in PS output
		  (the above thee plugins are Linux only, at the moment)
		- new quartz plugin (MacOSx only)
		- new gdiplus plugin (Windows only)
		- new -Tvml support in core plugin (thanks Steve Roush)
		- new -Ttk support in core plugin (also used by Tcldot and
		  gv_tcl language bindings.)
		- disabled old style codegens completely
	- Linux:
		- new Ubuntu8 builds
		- new Fedora 10 and 11 builds
	- MacOSx: 
		- Universal binary for Leopard: i386, x86_64, ppc, ppc64
		- Should not conflict with parallel install of MacPorts
		  version of graphviz
		- Improved GUI 
	- Windows:
		- VisualC project files now available, in addition to the GNU
		  Makefiles that are used the mingw builds.
	- Language Bindings:
		- fixed problem with writing dot, xdot, plain, canon to 
		  memory or to Tcl_Channels
		- renamed man pages to Debian style:  gv.3tcl, gv.3perl, etc
	- Fixed bugs: 827 1365 1366 1367 1368 1374 1375 1376 1378 1380 1382
	1383 1385 1386 1388 1390 1391 1392 1394 1395 1397 1398 1399 1405
	1407 1410 1412 1414 1415 1416 1421 1424 1425 1427 1429 1431 1433
	1435 1436 1437 1438 1440 1441 1444 1446 1451 1452 1453 1456 1457
	1459 1460 1461 1462 1463 1464 1465 1466 1470 1474 1475 1476 1477
	1478 1484 1485 1489 1490 1492 1493 1495 1496 1499 1500 1501 1502
	1503 1505 1509 1513 1521 1523 1525 1530 1531 1532 1533 1535 1536
	1539 1540 1542 1543 1546 1547 1551 1553 1554 1561 1565 1566 1568
	1569 1570 1571 1573 1577 1578 1579 1580 1581 1582 1584 1586
		
June 25, 2008
	- Release 2.20.2
	- Fix bug in HTML-like labels 
June 23, 2008
	- Release 2.20.1
	- Fix bug in ccomps related to conversion to cgraph
June 20, 2008
	- Release 2.20.0
	- Preparing for Release 2.20
	- Fixed bugs: 1315, 1317, 1324, 1336, 1343, 1364
	- Add new "folder" shape for nodes.
	- Migration of gvpr tools to libcgraph.   
	- New output format -Teps  (encapsulated postscript)
	- Various NetBSD and SuSE fixes incorporated
	- ./configure now provides a summary
	- RPM specfile updates for fedora-10 (no more string comparisons)
	- Add MacOS support (Glen Low)
March 10, 2008
	- Release 2.18
	- Fixed bugs: 1249 1255 1256 1268 1276 1289 1295 1300 
		Fedora BZ#247376, 
	- in -Tps use a new number formatter that suppresses trailing 0.
	- support tcl/tk-8.5
	- support gcc-4.3
	- support for node usershapes/images in svg format (thanks Alex Poylisher)
	- install: perl, php, python, ruby, tcl, bindings in language-specified directories
	- add arrowhead scaling with edge penwidth
	- add "folder" node shape (thanks Pander)
	- many windows and mac fixes (thanks Glen)
	- add "smyna" large graph view (thanks Arif) (not yet included in binary distros)
December 12, 2007
	- Release 2.16.1
	- Fixed bugs: 1228 1234 1238 1239 1245
	- Improvements to PHP binding
	- Improvements to OCAML binding
	- Make regression tests run from the build tree, rather than require installation
	- Repair freetype detection on RedHat-7 (Yes, people still use it!!)
	- Fix zoom-at-mouse-location in -Txlib and -Tgtk
	- Fix some dotty regressions
November 9, 2007
	- Release 2.16
	- Fixed bugs: 456 473 1021 1153 1154 1155 1159 1160 1162 1165 1166
	1168 1169 1170 1172 1173 1174 1175 1177 1178 1179 1181 1182 1183
	1185 1187 1189 1192 1193 1195 1196 1199 1204 1207 1210 1215 1216
	1217 1218 1219 1220 1223
	- new regression test suite
	- new cgraph library (will eventually replace graph and agraph)
	- add "image" and "imagescale" for simpler support for images in nodes
	- add "tab" "box3d" and "component" shapes.  - Diomidis Spinellis
	- replace arith.h in distro
	- add functions to access version info to avoid need for gvcint.h
	- Fix problem with irregular character spacing at 96dpi in pango/cairo output formats.
	- Add gdk_pixbuf plugin providing: .bmp .ico .jpg .png .tif 
	- Add DevIL plugin providing: .bmp .jpg .png .tif .tga
	- Extend GD plugin to provide a backend to cairo for: .gif .jpg .png .gd .gd2 .wbmp  <- gifs are now antialiased
	- Rework plugin framework to separate device from renderer, and to autoload load dependendent plugins
	- show defaults in output from: ./configure --help
	- add more info to dot -v  and dot -v2 debug outputs
	- various issues with CR/LF in windows, but not in binary outputs.
August 15, 2007
	- release 2.14.1
	- Fixed bugs: 1163, 1167
	- Windows build fixes
	- Add xdot parsing library to source distros
	- graphviz.spec fixes for rpm distros from Gareth Armstrong
	- moved language binding man pages to mann (gv_php.n, gv_ocaml.n, etc.)
	- New access functions for version info in GVC_t - permits gvcint.h to
	be private.
August 2, 2007
	- release 2.14
	- Fixed (or otherwise closed) bugs:
		74 130 162 184 190 197 219 223 281 295 311 316
		324 352 364 385 393 404 420 447 455 474 489 507
		530 532 537 543 551 564 571 574 577 583 587 588
		590 592 595 599 638 647 650 660 675 667 668 669
		676 684 685 686 721 725 734 740 746 747 748 749
		752 755 756 765 778 780 781 782 785 794 803 814
		822 828 836 840 847 852 862 866 868 893 928 944
		948 950 955 961 976 985 992 1024 1057 1064 1065
		1066 1069 1072 1074 1079 1085 1086 1089 1091 1092
		1093 1094 1096 1107 1111 1123 1124 1130 1138 1145
		1151 1152 1156
	- Fixed Redhat bugs: 218191 237497
	- Fixed Debian bugs: 321128 422862 422873
	- Fixed Gentoo bugs: 173676
	- Using system version of libgd if gd-2.0.34 or later. (Fedora 7 and 8 distros)
	        internal copy of gd updated to gd-2.0.35.
	- Updated GVGUI viewer for Windows
	- Windows build process now uses GNU autoconf and UWIN
	- Added support for selection of edge routing types:
		line, polyline, orthogonal, spline
	- Added -Tvml support
December 5, 2006
	- release 2.12
	- Bug fix release for 2.10
	- The gd plugin for font handlers was not being used at all if the build
	did not use fontconfig, e.g., on Windows. In addition, the code had
	dropped the name mapping to Windows font names.
	- PostScript output had an extraneous '%' character on the first line,
	which would cause printing to fail.
	- Text handling, during both sizing and layout, incorrectly handled
	empty lines such as label="\nabc".
	- HTML-like tables had been changed to use too much vertical space,
	to possibly use the wrong font in calculating the height of a line,
	and to use the wrong offset when moving the baseline from one line to
	the next.
November 27, 2006
	- release 2.10
	- dot - New pango+cairo renderer plugin (was in separate graphviz-cairo tree).
	  -- -Tpng now uses cairo   (-Tpng:gd for old gd based renderer)
	  -- -Tpdf now available
	  -- -Tps:cairo now available (-Tps is a direct ps renderer not based on cairo)
	  -- -Tsvg:cairo now available (-Tsvg is a direct svg renderer not based on cairo)
	  -- -Txlib now available -- "dot -Tx11 foo.dot"  watches foo.dot with inotify and updates
	  -- -Tgtk now available -- eventually to provide a graph editing capability - not fully working
	  -- -Tswf "Flash" now available using the ming library. Currently has incomplete font support and not yet in Fedora rpms because ming not yet available as rpm.
	- remove hard gd dependencies from dot.  gd renderers now provided
	  as optional plugin.   Deprecated, but required for -Tjpg, -Tgif and -Tvrml.
	- gvpr - Add kindOf function, plus functions to set and get default values
	- dot - Implement esep attribute to allow graph to specify room
	around nodes for spline routing.
	- neato - add vpsc library and DIGCOLA
	- neato - add IPSEPCOLA additions from Tim Dwyer
	- move: -Tps, -Tfig, -Tsvg, -Timap/ismap/cmap/cmapx, -Tdot/xdot,
	from codegens to a "core" plugin.
	- dot - new usershape plugin mechanism potentially supporting
	  a wider range of input shape format -> output format combinations.
	display on changes
	- Fixes for builds on Mac OS/X
	- dot - new -O switch to automatically generate output file
	names based on the input filename and the -T value.
	 e.g.  "dot -Tpng -O *.dot"   
	Also works for case of multiple graphs in a single input file.
	- add support for "Brewer" color nameset
	- move reusable .so libraries to $PREFIX/lib per frequent request
	from Debian community.   Plugin .so's remain in $PREFIX/lib/graphviz.
	- Fix bugs 882 884 886 896 902 905 906 911 918 919 933 936 938 940
	   948 955 958 967 979 987 993 1005 1006 1011 1012 1013 1014 1016
	   1018 1025 1030 1034 1035 1039 1040 debian#37300

February 3, 2006
	- release 2.8
	- (POTENTIAL INCOMPATIBILITY) The default input scaling, in the
	absence of a "-s" switch, has been changed from inches to points.
	The new behavior of "neato" is equivalent to "neato -s72".
	The old behavior can be restored with "neato -s1".
	The purpose of this change is to avoid a Frequently-Made-Mistake
	when using "neato -n" to process a previously generated layout.
	Previously it was necessary to use "neato -n -s72", but with this
	change the default matches dot's output and the "-s72" is not required.
	- Added pseudo layout engines: "dot -Knop" and dot -Knop1" equivalent
	to "neato -n"
	- Added pseodo layout engine: "dot -Knop2" equivalent to "neato -n2"
	- Add support for color namespaces; add Brewer color data
	- Add support for simulated duplex edges using parallel edges:
	head arrow takes first color, tail arrow takes second color.
	- source code management moved back to CVS until GIT matures a bit more
	- distribute separe rpms for binares of language bindings : 
	- Add a small pad region around graph renderings to allow for finite
	penwidths at the drawing edges
	- Add protonode(g) and E=protoedge(g) functions to simplify
	language bindings.
	- Add special purpose code to deal with html labels from language
	bindings.
	- Various portability fixes for: HPUX, Mac OS/X, Cygwin, Windows.
	- Fix bugs 784 786 787 788 789 790 791 793 795 796 798 799
	    800 801 804 806 811 812 817 820 821 823 824 825 830
	    837 839 841 842 843 848 850 851 854 855 856 857 858
	    859 861 863 866 867 869 872 874 876 877

August 28, 2005
	- release 2.6
	- experimentally moved source code management from CVS to GIT
	- added iterator functions to script bindings
	- more C-API tuning
	- add "-c" switch to dot to explicitly generate plugin "config" file
		instead of generating it as a side-effect of "dot -V"
	- better support for binary relocation.
	- plugin versioning and version checking
	- clean up of header files
	- provide statically linked "dot_static" (not incl. in rpms)
	- additional "event" support for GUIs (e.g. "DotEdit" graphviz-cairo)
	- add some information about plugins to "dot -v" output.
	- lefty/dotty fixes
	- fix bugs 746 750 752 753 754 756 761 763 764 765 768 
		771 772 773 774 775 776 777 778
	- not a bug 757 760 770
July 20, 2005
	- release 2.4
	- major code restructuring
	- new plugin architecture (e.g. see separate package: graphviz-cairo )
	- new script-language bindings using swig (perl, ruby, python, tcl, java ... )
	- C-API now in libgvc (no more dotneato.[ch] or dotneato-config.sh]
	- pkgconfig now used for reusable libraries
	- lefty upgrade
	- fix bugs 156 255 492 631 641 647 659 662 665 670 690 691
			701 702 703 705 730 731 732 741 743
April 7, 2005
	- release 2.2.1
	- correct license headers to CPL in .cpp files
	- undo indentation cleanup to dynagraph .h files
	- fix bugs: 183 247 419 615 616 625 626 627 643
		646 651 658 661 664 674
	- fix buffer overrun in Gvfilepath construction
January 19, 2005
	- release 2.2
	- fix bugs: 86 345 517 579 580 597 600 601 604
	- use the original cpl1.0.txt as the license master, instead of CPL.html        - fix for bug generating in memory bitmaps that was affecting webdot
	- fixes for windows builds
	- documentation updates
December 11, 2004
	- release 2.0
	- new CPL license
	- re indent all sources
December 11, 2004
	- release 1.18
	dotneato
	- fix bugs: 451 536 545 547 548 559 561 565 572
	- increase max size  of HTML tables.
	- spline cluster edges in fdp
	- center userimages in nodes
	- support user images in HTML table cells
	- syntax extension for node:port:compass as well as node:compass
	- FreeBSD fixes
	- sync with gd-2.0.32
	- attempt to catch some out-of-memory conditions with very large graphs
	- support background and node-fill partial transparency when truecolor=true
		
September 14, 2004
	- release 1.16
	dotneato
	- fix bugs: 275 523 526 527 529 534
August 30, 2004
	- release 1.14
    dotneato
	- the official gd now has support support for GIFs again - the
		internal gd is now closely sync'ed with the official version
		and will eventually be removed in favor of using a
		separate installation of the official version.
	- gd has new support for FontConfig (thanks to Dag Lem)
		NB. the fontname attribute in graphs is now a font pattern
		as understood by fontconfig (e.g. fontname="Times-Italic"),
		unless it contains a '/' in which case it is interpreted as
		a font path as before.
	- gd provides support for html4 entities in decimal, hex or named, e.g "&lt;"
	- "dot -v" debugging output now reports fontname -> fontpath resolutions

	- PostScript generated by -Tps now uses "xshow" operator for strings
		for better matching of bitmap and PostScript outputs.

	- ability to use an external gd-2.0.29 version of libgd (EXPERIMENTAL)

	- new feature: parallel edges by using a ":" separated list of edge colors
	- new feature: rankdir=BT and rankdir=RL  (thanks to Dag Lem)

	- new layout engine: fdp - force directed placement (EXPERIMENTAL)
		a neato-like undirected layout engine that produces
		clustered symmetric layouts.
		Supports edges between clusters and nodes.

	- updated neato engine: now using stress majorization as the default,
		which avoids the potential for cycling
	- model=subset in neato provides a third distance function, where
		two nodes sharing many nodes will be place farther apart
	- shape=none now equivalent to shape=plaintext
	- fix label justification with \l and \r
	- first cut at <FONT> support added to html labels
	- various color transparency fixes
	- various fixes for UTF8 and Latin[12] character encodings.
	- various cluster fixes.
	- improved hyperlink support in -Tsvg
	- support tooltips on clusters in client-side imagemaps

    gvpr
	- add support for scanf and friends

    general
	- greater use of shared libraries.
	- pkg-config files provided for shared libraries (EXPERIMENTAL)
	- "./configure --disable-shared --enable-static" works if needed
	- C++ wrappers on all header files (thanks to Victor Wodecki)
	- various configuration and portablity fixes
	- provide pdf version of man pages
	- Windows package provides graphviz libraries and header files
	- Closed bugs: 195 198 234 321 330 399 401 406 410 411
		412 413 415 416 417 423 424 427 430 431 433 434 435
		438 441 442 444 445 449 450 452 454 457 458 462 463
		464 467 468 469 471 475 480 482 485 495 496 498 499
		500 501 504 508 511 512 514

March 5, 2004
    - added glyphwidths.ps support utility

March 1, 2004
    - release 1.12
    - general
	- rename bcc -> bcomps to avoid name conflict with "Bruce's C Compiler"
		on Redhat distributions.
	- all build without X11 (fix problem in lefty tree)
	- remove from distribution:
		dag, fdp, geo, grid, incr, shape, tcldgr, tcldgl
    - dotneato
	- fix "brown-bag" problem resulting in PNG and JPEG errors on RH8 and RH9.
February 23, 2004
    - release 1.11
    - general
	- fix windows builds
	- add tool "bcc" to distribution
    - dotneato
	- add -Gviewport="X,Y,Z,x,y"  where XY are the dimensions of a viewport
	  in device coordinates (pixels), Z is a zooming factor, x,y is the
	  location of the center of the viewport in graph coordinates.
	  Supported in bitmap and imagemap outputs only.
	- fix memory leak in gd/gdft.c
	- clean up calculation of whitespace around labels
    - dotty, lefty
	- fix for bug #400	
December 23, 2003
	- added dijkstra (single source distance) filter
September 10, 2003
    - general
	- removed CVS directories from .tar.gz distributions
	- add "config" directory to contain some of the autoconf clutter
	- only remove flex products with "make maintainer-clean" to
	  avoid trying to regenerate them after "make distclean"
	  basically this is to avoid the broken flex on Debian.
	- suppress complaints from ./configure about config.rpath
	- doc/build.html updated with notes about Windows builds
	- build fixes for Forte 6sp2 compiler on Sun -xarch=v9a (64bit)
	- build fixes for OpenBSD
	- improved configure testing for Tcl/Tk
	- various bug fixes, internal restructuring, etc
    - dotneato
	- fix problem with extra escape chars in .fig output
	- support for "setlinewidth" in -Tfig
	- improved splines in -Tfig
	- add manpage for dotneato-config
	- neato: add defaultdist graph attribute to set distance
	  between components
	- first cut at html table formatter add. not ready for use yet
	  as the syntax is going to change some more.
    - tools
	- renamed "colorize" to "gvcolor" to avoid conflict on Debian
	- renamed "gpr" to "gvpr" to avoid conflict on Debian
	- add fflush() to acyclic, ccomps, gvcolor, tred, dot2gxl
	  to try to fix truncated output when used in php or perl cgi scripts
July 9, 2003
	- rerelease 1.10 with ast_common.h fix in -devel rpms
July 3, 2003
	- declare this version 1.10
	- general
	    - "mkdir obj;cd obj;../configure;make"   now works (bug #293)
	    - "make prefix=xxx"   now works (bug #274)
	    - "--with-wish=xxx"   now works (bug #270)
	    - remove generated file: ast_common.h from source distributions
	    - make GIF support configurable
	    - added .cvsignore throughout source tree to reduce CVS noise
	    - FAQ updates
	    - documentation updates for gpr
	    - improve portability of dotneato-config, but requires libtool now
	    - improvements to error processing for library users
	-gd
	    - sync with gd-2.0.15
	    - optimize line drawing code
	- dot, neato, twopi
	    - fix bugs 240 270 274 293 298 303
	    - support "peripheries=0" without crashing
	    - add support for "dia" output format (-Tdia)
	    - espf fixes (use of showpage)
	    - svg fixes (coordinates and viewBox)
	    - ismap/imap, fixes (quoting of label strings)
	    - fix to "point" shape
	    - improve (m|c|re)alloc usage
	    - improve handling of very-small fonts in bitmap outputs.
	    - various fixes for multiple -T -o feature
	    - add support for splines to records and ports (neato)
	    - various improvements to libpack
	    - dot_init_graph and neato_init_graph external for library users
	    - cluster improvements (neato)
	    - fix support for truecolor
	    - normalize splines so that they now always go from tail to head
	    - add some simple help text for any unrecognized option
		(e.g. -?  -h  --help)
	- tools
	    - extend gpr language to allow access to command-line arguments
	    - add sqrt() function to gpr
	    - add new tool - gvpack
	- tcldot
	    - use .dll extension if on windows
	    - doted demo
		- use tcl's file requestor instead of homebrew
		- add zooming controlled by mousewheel
		- support additional export formats
	    
January 31, 2003
	- declare this version 1.9
		(3-level version numbering has been dropped now
		that we have nightly snapshot builds with their
		own extended numbering.)
	- general
	    - config.h is no longer installed.  config.h is generated by
		./configure for the current build only.  It may not be
		applicable for derivative builds.
	    - improve ICONV configure tests
	    - lots of janitor-work to clean up warning messages from -Wall
	    - use @OBJEXT@ in Makefile.am so that .obj is used under cygwin
	    - fixes for Solaris builds
	    - use libpng-config if available
	    - reduce long build times due to touching ast_common.h too often
	    - improve dependency tracking.  "make -j8" now works with distcc
	    - autogen.sh fixes to work on RH7.3, RH8.0, and Solaris.
	    - eliminate use of suffix rules which confused some makes.
	    - DOT language allows '+' for concatenation of quoted strings
	- dot, neato, twopi
	    - fix bugs 209 210 214 216 217 222 224 225 229
			230 233 236 237
	    - update gd into alignment with gd-2.0.9
	    - change to make libagraph output compatible with libgraph input
	    - add shapes: septagon, pentagon, a_ediamond, rect, rectangle
	    - introduce "ND_...", "ED_...", "GD_...", node/edge/graph-data
		accessor macros in partial preparation for use of
		libagraph in dot.
	    - add libdotneato.so, dotneato.h, dotneato-config
		to aid use of dot libraries by user apps based
	        on installed graphviz-devel rpm and without access
		to graphviz sources.
	    - new xdot output format providing detailed drawing instructions
	    - new -y command line flag, inverts y coordinates
	    - support multple -T when -o given, as in:
			cat xxx.dot | dot -Tpng -Tcmap -o xxx
		which produces xxx.png and xxx.cmap from a single
		layout computation.   Intended for use in CGI programs.
	- agraph
	    - correct callback ordering for deletions
	- tools
	    - add gxl2dot and dot2gxl for GXL language conversions
	    - gvui now provides *map output
	- tcldot, tcldgr, tcldgl
	    - improve tcl8.4 support
	    - extend search path for tcl.h to include /usr/local/include/tcl8.4/
		in support of BSD install conventions.
	- dynagraph
	    - many fixes
	    - change to not build dynagraph by default (use --with-dynagraph)
	- docs
	    - dotguide updates
September 27, 2002
		- declare this version 1.8.10
	- general
	    - various configure.in fixes and simplifications
	    - change configure to now build dynagraph by default
	    	"--without-dynagraph" is supported
	    - fix graphviz.spec.in to partition packages properly
	    	graphviz no longer depends on graphviz-tcl.
	    -  Makefile.old cleanups
	    - configure.old now set version number automatically from
	      configure.in
	- dot, neato, twopi
	    - Initial support for image node shapes + URL fetch.
	    - Made number of dimensions a runtime variable in neato.
	    - Bug fix in vrmlgen for degenerate splines.
	    - Bug fix - ordering=in should now work
	    - Bug fix - layers no numbered from 0 to match PS requirements
	    - Bug fix - don't draw arrows on invisible edges
	    - Bug fix - when pack=true and ratio is set
	    - Bug fix - agraph/scan.l to work with latest flex beta

August 2, 2002
		- declare this version 1.8.9
	- general
	    - split rpm into:
	        graphviz, graphviz-tcl, graphviz-graphs, graphviz-devel
	    - gcc3 warning cleanup
	    - Install lincdt, libgraph, libagraph, libgd, libpathplan, libexp,
	    	and libpack so that they can be used by other programs. 
		Headers and man3 in graphviz-devel
	- dynagraph, graphsearch
 	    - New tools based on libagraph and written in C++
	- dot, neato, twopi
	    - Add node and edge tooltips for use with -Tcmap
	    	\N,\E,\H,\T substitutions also work in tooltips.
	    - Add alt="label_string" to -Tcmap
	    - Add edge-label and port mappings to -Tps and -Tps2 so
	        that edges can be hyperlinked in PDF documents.
	    - Add support for \E (edge name), \H (head-node name),
	        \T (tail-node name) substitutions in edge labels and edge URLs
	    - Add support for stylesheet="file.css" for use in -Tsvg
	    - Fix -Tpic to work with recent gpic (Bruce Lilly)
	    - Fix alignment of imagemaps to images.
	    - Fix "transparent" color support in -Tsvg
	    - Fix support for graph [URL="default.html"] in -Tsvg and -Tcmap.
	    - Fix '&' escaping in URLs in -Tsvg
	    - Fix infinite loop in dot layout algorithm
	    - Fix text rotations again (hopefully freetype is stable now.)
	    - Cluster layout improvements
	    - Clean up warning messages from pathplan
	    - Consolidation of mapping code from imapgen.c and ismapgen.c into mapgen.c
	- gpr
	    - Added additional mode to extract components based sharing an
	        edge or a cluster
	    - Fix test for getopt
	- tcl-based tools
	    - Disable tcl-based tool building if tcl/tk not available
	        with stubs support.
	- documentation updates: FAQ, dotguide, dot.1
July 5, 2002
	    - declare 1.8.7 a "brown bag" release
		 and declare this version 1.8.8
	- remove wrong assert in gdgen.c
	- fix graph centering in bitmap outputs
	- provide enough margins
	- fix line widths after scaling 
		(test with directed/proc3d.dot)
	- fix text rotations (requires libfreetype.so.6.3.1) 
		(test with directed/NaN.dot)
July 5, 2002
	    - declare this version 1.8.7
	- Fix missing "]" in ihi demo.
July 2, 2002
	- Add URL mappings for clusters: svg,svgz,ps,ismap,imap,cmap.
	- Fix to avoid white edges in bitmap outputs when bgcolor is set.
	- Improve sizing and position of strings in bitmap outputs
	  when using builtin fonts (when font file not found).
	- Fix \N substitution in edge URLs in imap and cmap outputs.
	- Add -Tcmap for client-side imagemaps.
	- Generate warnings instead of access violation for EPSF file problems.
	- Various spline fixes in neato.
	- Fixes to pack.c
	- Add feature to ccomps to allow extraction of individual component
	  by number or node.
	- Cdt make to use iffe provided in the tools directory.
	- Various Makefile.old fixes.
	- Use HAVE_LIBZ to remove GD2 format if libz not available.
	  Now bare-bones programs can be built without any add-on libraries.
	- Modified dot grammar to allow simple name attributes in attribute
	  lists.  Thus, [splines] is equivalent to [splines=true]. Adopted
	  the same convention for command line attributes -G, -E and -N.
	  In addition, such command line attributes now override any
	  competing initial attribute statements.
	- HP-UX 11.11 build fixes for struct dioattr.
	- Fix for bug #158 "Nodes disappear with ports"
	- Various Windows-specific #ifdefs
	- Fix edge coordinates in -Tplain.
	
May 24, 2002
	    - declare this version 1.8.6
May 19, 2002
	- Fixed segfault from use of bgcolor in clusters.
May 15, 2002
	- Changed install location of architecture-independent demo
	  scripts and graphs to <prefix>/share/graphviz/ to conform to FHS.
	- Avoid multiple linking of libfreetype (and others) which caused
	  problems on SunOS-2.8.
May 6, 2002
	- Factored out some duplicated arrow code from dotgen/splines.c
	  and neatorgen/splines.c into common/arrows.c.
	- Added new arrow types:  halfopen, box, obox, crow.
	- Touched up the arrow designs so that they look better at default size.
	- Modified/extended graphs/directed/newarrows.dot to show new arrows.
May 3, 2002
        - Added some UML arrow types from Diomidis Spinellis <dds@aueb.gr>
	  empty, invempty, open, diamond, odiamond.
May 2, 2002
	- Added new pack option to neato. This causes each connected component
	  to be laid out separately, and then the resulting graphs are packed
	  together in a single layout.
	- Amended neato to accept new tee arrowhead.
April 19, 2002
	- Coords of rectangles changed to left/top right/bottom in -Timap.
	- Generate COPYING from LICENSE.html during ./authogen.sh,
	  remove COPYING from CVS.
April 16, 2002
	- Minor license file patches.
	- Corrected one of those reversed flat edge bugs again.

April 11, 2002
	     - declared this version 1.8.5
	- various portability fixes 
	- various SVG fixes and optimizations
April 5, 2002:
	     - declared this version 1.8.4
	- SVG renderer:
		- make graph|node|edge ids unique, particularly for multiedges
		- put graph|node|edge names in <title>...</title>
		- use some property inheritance to reduce size of output
		- fix compile errors when no zlib
		- updated DTD reference
	- GD renderer:
		- Minimal Type1 font support:
			- look in /usr/lib/X11/fonts/Type1/
			- look for .pfa or .pfb font files based on fontname
		- run gdgen.c through dos2unix - problems with gcc on SuSE
	- fix Mac-OSX build problems:
		- improve strto[u]ll configure tests
		- add -fno-common for extern problem
		- function renamed to avoid conflicts (vis -> visibility)
		- add configure tests for search.h, malloc.h, getopt.h, errno.h
		- improve configure tests for FILE struct features
		- add configure tests for lrand48
	- add new demo graphs:
		- graphs/undirected/Heawood.dot
		- graphs/undirected/Petersen.dot
	- neato:
		- fix for -x implementation in neato (Bug 77)
		- fix spline problem (Bug 87)
		- fix some divide-by-zero problems
	- twopi:
		- fix Bug 117
		- update man pages for unconnected graphs capability
	- added arrowhead or arrowtail = tee
March 22, 2002:
	- add dotneato/pack code to twopi
	- add contrib/prune to gnu build and install
March 20, 2002:
	    - declared this version 1.8.3
	- fixed parse error for lines starting with '#' in .dot files
	- fixed a recently introduced bug that caused failure of:
		digraph G {  {rank = same;  A -> B; B -> A } }
	- updated DOCTYPE header in SVG outputs
	- added dotneato/common/xbuf.[ch] for dynamic string handling
	  to avoid sprintf buffer overruns.
	- twopigen - handle special case of graphs with < 3 nodes.
	- neato - handle point shapes
	- added fontcolor support to svg
March 14, 2002:
	- Fixed bug 109
	- Removed duplicate definitions for str[n]casecmp
	- Added missing declarations needed for Windows
	- Cleaned up warning messages from set but unused variables
	- Removed use of DOS preprocessor variable; uniformly replaced by MSWIN32
March 8, 2002:
	- declared this version 1.8.2
    - Mainly to fix a missed static buffer problem which trips up the
      Windows community
March 1, 2002:
	- declared this version 1.8.1
    - Bug fixes reported from user testing of 1.8.0, especially problem
      with SVG output
February 25, 2002:
	- updated dotguide.tex and moved to LaTeX article format
	- added webdot.cgi perl script, enhanced to accept the same
	    argument format as John's tcl version (so it can also
	    serve neato and twopi graph layouts).

February 7, 2002: graphviz-1.8.0 pre
	- declared this version 1.8.0

February 5, 2002: graphviz-1.7.17-0
    - various 64bit portability fixes
    - various bug fixes
January 2, 2002: graphviz-1.7.16-0
    - dotneato 
	- fix bugs in -Tps output due to pen/fill color changes
	- various -Tfig.c fixes
	- various portability fixes
December 28, 2001: graphviz-1.7.15-0
    -dotneato
        - introduce damping factor into neato's solver
        - clean up pencolor v fillcolor code so that filled polygons are drawn
		just once if the renderer is capable (e.g. svg, fig)
        - complete -Tfig support (xfig format)
December 11, 2001: graphviz-1.7.14-0
    -dotneato
	- add -Tsvgz (compressed SVG) support
December 11, 2001: graphviz-1.7.13-0
    - dotneato
        - fontwidth fixes
	- remove some potential buffer overruns
	- escape '&' in SVG, unless it is already part of a UTF entity sequence
	- recognize Times_New_Roman and Courier_New as default font names.
	- improve -liconv support in configure
	- clean up some compiler warnings
    - dynagraph
	- change "round" to "ROUND" to avoid conflict with system headers on linux
December 03, 2001: graphviz-1.7.12-0
    - dotneato
        - add -Tplain-ext which includes port identifiers edge records
	- escape '>' with '&gt;' in edge ids and edge URLs in -Tsvg.
	- spline fixes
	- mincross fixes
	- improved text alignment in nodes - particularly in bitmap outputs.
	- fixed text scaling problems for 8-bit characters (e.g. umlauts)
	- add graph lexer and postscript support for extended characters
    - lefty
        - fix for X11 displays
    - pathplan
        - added workaround for gcc-0.96 bug when "-O2 -mcpu=686 -ffast-math"
October 22, 2001: graphviz-1.7.11-0
    - dotneato
	- svg - fix landscape "y" direction
	      - fix text rotation (works in batik, not yet in sodipodi or amaya)
	      - fix linewidth
	      - fix xmnls:xlink reference
    - doc
	- Dot.ref - updated 
    - graphs/directed
        - newarrows.dot expanded 
	- honda-tokoro.dot added
October 21, 2001: graphviz-1.7.10-0
    - lefty & dotty
	- realign code with EK's master tree.
	  includes fix for dirty trails when dragging nodes in dotty.
    - dotneato
	- svg - kludge escape of "<" & ">" characters in labels.
    - general
	- generate doxygen documentation on http://www.graphviz.org/
August 20, 2001: graphviz-1.7.9-0
    - general
	- first release from relocated cvs server
    - dotneato
        - fix for abort from spline code
        - fix for crash from gd tiling code
August 15, 2001: graphviz-1.7.8-0
    - general
        - Update gd to gd-2.0.1 with extensions
    - dotneato
        - more spline fixes
        - add suport for "#rgb" color specification
        - add twopi layout engine (circular layouts)
July 13, 2001: graphviz-1.7.7-0
    - Synchronization release prior to relocating CVS server.
    - general
    	- some Makefile fixes for OpenBSD
	- some FAQ updates
    - dotneato
        - self-edge fixes
        - spline fixes
    - libgraph
        - parser fixes
July 1, 2001: graphviz-1.7.6-3
    - general
	- portability fixes (including 14 charater file names !)
	- memory leak fixes
	- "make test" targets in graphs/directed, graphs/undirected
    - configure
	- add support for building without X11, Tk, Tcl
	- add hooks for dmalloc and ElectricFence debugging
    - dotneato
	- spline fixes
	- cluster fixes
	- fix label centering
	- fix support for graph margins in bitmapped outputs
	- correction to PostScript preamble
	- SVG generator improvement - now works with Amaya and SodiPodi
    - tcldot 
	- now uses Tcl Channels properly for input
	- fixes for linewidth support
	- command extensions 
	    - listattributes now accepts list
	    - queryattributes now accepts list
	    - setattributes now accepts list
	    - queryattributevalues - new command
		- generates list of pairs compatible with setattributes
    - dotty
	- passthrough keyboard events
    - doted
	- fix resizing problems
	- add PNG and SVG output formats
 
April 27, 2001: graphviz-1.7.6

    NEW FEATURES

    Added a collection of graph processing tools:

    acyclic : a filter that takes a directed graph as input
    and outputs a copy of the graph with sufficient edges
    reversed to make the graph acyclic.

    ccomps : decomposes graphs into their connected components,
    printing the components to standard output.

    colorize : is a filter that sets node colors from initial
    seed values. Colors flow along edges from tail to head.

    gc : a graph analogue to wc in that it prints to standard
    output the number of nodes, edges, connected components or
    clusters contained in the input files.

    gpr : a graph stream editor inspired by awk. It copies
    input graphs to its output, possibly transforming their
    structure and attributes, creating new graphs, or
    printing arbitrary information.

    nop : reads a stream of graphs and prints each in
    pretty-printed (canonical) format on stdout.

    sccmap : decomposes digraphs into strongly connected components
    and an auxiliary map of the relationship between components.

    tred : computes the transitive reduction of directed graphs,
    and prints the resulting graphs to standard output. This
    removes edges implied by transitivity.

    unflatten : is a preprocessor to dot that is used to improve
    the aspect ratio of graphs having many leaves or disconnected
    nodes. The usual layout for such a graph is generally very
    wide or tall. unflatten inserts invisible edges or adjusts
    the minlen on edges to improve layout compaction.


    FIXES

    Add FAQ

    Change PNG default background color from transparent to white
    because of the difficulty some viewers have with transparency.

    Add support for [color=transparent]

    Fix broken support for specific capitalized fontnames
    (Times Helvetica Arial Courier) 

    Fix broken support for DOTFONTPATH

    Some bitmap font scaling fixes - we're still not happy with
    bitmap font scaling as some labels still exceed the area
    allocated by the layout engines.

    Some -Timap fixes for mouse sensitive graphs on web pages

    Some cluster layout fixes

    Fix for [rankdir=LR] problems when using neato layout engine

    Some neato layout fixes

    Updates to unix.dot

    Various OS and distro fixes


December 23, 2000: graphviz-1.7.5

   - update to gd-1.8.4 and freetype2 
   - add support for font paths


December 15, 2000: graphviz-1.7.4
    -various cluster fixes
    -separate support for node fillcolor from pencolor (see dot.1)
    -add support for dotted and dashed lines to bitmap renderers (PNG, GIF etc)
    -add support for varying linewidth to bitmap renderers
    -remove libtcldot dependence on lingdtclft (already statically included)
    -various fixes to build processes, GNU and non-GNU


graphviz-1.7.3 .....

May 3, 2000: removed webdot into its own CVS module and rpm package

April 16, 2000: Use check for "gdImagePng" to make sure that we have
   recent version of libgd.  <ellson@graphviz.org>

April 14, 2000: Add Tcldgl and dge demo <ellson@graphviz.org>

April 14, 2000: Add dynagraph libraries <north@research.att.com>

April 14, 2000: Flatten directory hierarchy of sources <ellson@graphviz.org>

April 14, 2000: Fix X11 library detection for lefty:
	src/configure.in, src/lefty/Makefile.in
   <ellson@graphviz.org>

April 14, 2000: Fix pic support:
	src/dotneato/picgen.c,
	src/dotneato/emit.c,
	webdot/tcl/webdot.tcl
   <Bruce Lilly>

April 7, 2000: Upgrade webdot installation process:
	webdot/Makefile, webdot/README
    <ellson@graphviz.org>

March 13, 2000: Support for virtual hosts in webdot/webdot.tcl, add
   "puts $skt "Host: $server"     Michael Tillberg <mt@proteome.com>

March 13, 2000: Fix to src/graph/parser.y line 149
   "if ((e->head == t->node) && !(Agraph_type & AGDIGRAPH)) {"
   Stephen North  <north@research.att.com>

March 13, 2000: Use AM_PROG_LIBTOOL instead of AC_PROG_LIBTOOL
   in configure.in.  John Ellson <ellson@graphviz.org>
```

[2.50.0]: https://gitlab.com/graphviz/graphviz/compare/2.49.3...2.50.0
[2.49.3]: https://gitlab.com/graphviz/graphviz/compare/2.49.2...2.49.3
[2.49.2]: https://gitlab.com/graphviz/graphviz/compare/2.49.1...2.49.2
[2.49.1]: https://gitlab.com/graphviz/graphviz/compare/2.49.0...2.49.1
[2.49.0]: https://gitlab.com/graphviz/graphviz/compare/2.48.0...2.49.0
[2.48.0]: https://gitlab.com/graphviz/graphviz/compare/2.47.3...2.48.0
[2.47.3]: https://gitlab.com/graphviz/graphviz/compare/2.47.2...2.47.3
[2.47.2]: https://gitlab.com/graphviz/graphviz/compare/2.47.1...2.47.2
[2.47.1]: https://gitlab.com/graphviz/graphviz/compare/2.47.0...2.47.1
[2.47.0]: https://gitlab.com/graphviz/graphviz/compare/2.46.1...2.47.0
[2.46.1]: https://gitlab.com/graphviz/graphviz/compare/2.46.0...2.46.1
[2.46.0]: https://gitlab.com/graphviz/graphviz/compare/2.44.1...2.46.0
[2.44.1]: https://gitlab.com/graphviz/graphviz/compare/2.44.0...2.44.1
[2.44.0]: https://gitlab.com/graphviz/graphviz/compare/2.42.4...2.44.0
[2.42.4]: https://gitlab.com/graphviz/graphviz/compare/2.42.3...2.42.4
[2.42.3]: https://gitlab.com/graphviz/graphviz/compare/2.42.2...2.42.3
