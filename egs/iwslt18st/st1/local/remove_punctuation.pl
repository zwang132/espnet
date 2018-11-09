#!/usr/bin/perl

use warnings;
use strict;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

while(<STDIN>) {
  $_ = " $_ ";

  # normalize punctuation
  s/&/ & /g;
  # s/，/,/g;
  # s/‒/-/g;
  # s/‟/"/g;

  # s/ʹ/'/g;

  # reserve some punctuations
  s/'/apostrophe/g;
  s/&/ampersand/g;
  s/#/numbersign/g;
  s/\$/dollarsign/g;
  s/%/percentsign/g;
  s/\+/plussign/g;

  # remove punctuation
  s/[[:punct:]]/ /g;
  # 一応何消したかわかるように残す

  # reserve some punctuations
  s/apostrophe/'/g;
  s/ampersand/&/g;
  s/numbersign/#/g;
  s/dollarsign/\$/g;
  s/percentsign/%/g;
  s/plussign/\+/g;

  # remove whitespace
  s/\s+/ /g;
  s/^\s+//;
  s/\s+$//;

  print "$_\n";
}
