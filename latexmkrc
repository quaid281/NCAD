# latexmkrc for Overleaf: compile both supplement and standalone to resolve xr / xr-hyper references
$pdf_mode = 1;
if (-f 'NCAD_CS_supplement.tex') {
    push @default_files, 'NCAD_CS_supplement.tex';
    push @default_files, 'NCAD_CS_standalone.tex';
} elsif (-f 'paper/NCAD_CS/NCAD_CS_supplement.tex') {
    push @default_files, 'paper/NCAD_CS/NCAD_CS_supplement.tex';
    push @default_files, 'paper/NCAD_CS/NCAD_CS_standalone.tex';
}
