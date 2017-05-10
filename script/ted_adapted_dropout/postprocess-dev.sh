#/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/vol/soft-tlp/translate/moses/3.0/m15x/

# suffix of target language files
lng=cs

sed 's/\@\@ //g' | \
$mosesdecoder/scripts/recaser/detruecase.perl
