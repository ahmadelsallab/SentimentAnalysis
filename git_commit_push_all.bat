git submodule foreach git add *
git submodule foreach "git commit -m'%1' || :"
git submodule foreach git push origin HEAD:master
git add *
git commit -m"%1"
git push origin master:master