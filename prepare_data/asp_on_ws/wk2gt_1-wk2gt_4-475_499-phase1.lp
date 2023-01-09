
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(wk2gt_1,125,475). coords(wk2gt_1,115,53,475).  visible(wk2gt_1,475). happensAt(inactive(wk2gt_1),475). orientation(wk2gt_4,149,475). coords(wk2gt_4,77,98,475).  visible(wk2gt_4,475). happensAt(walking(wk2gt_4),475).
orientation(wk2gt_1,125,476). coords(wk2gt_1,115,53,476).  visible(wk2gt_1,476). happensAt(inactive(wk2gt_1),476). orientation(wk2gt_4,149,476). coords(wk2gt_4,76,97,476).  visible(wk2gt_4,476). happensAt(walking(wk2gt_4),476).
orientation(wk2gt_1,125,477). coords(wk2gt_1,115,53,477).  visible(wk2gt_1,477). happensAt(inactive(wk2gt_1),477). orientation(wk2gt_4,149,477). coords(wk2gt_4,75,97,477).  visible(wk2gt_4,477). happensAt(walking(wk2gt_4),477).
orientation(wk2gt_1,125,478). coords(wk2gt_1,115,53,478).  visible(wk2gt_1,478). happensAt(inactive(wk2gt_1),478). orientation(wk2gt_4,146,478). coords(wk2gt_4,75,97,478).  visible(wk2gt_4,478). happensAt(walking(wk2gt_4),478).
orientation(wk2gt_1,125,479). coords(wk2gt_1,115,53,479).  visible(wk2gt_1,479). happensAt(inactive(wk2gt_1),479). orientation(wk2gt_4,146,479). coords(wk2gt_4,75,97,479).  visible(wk2gt_4,479). happensAt(walking(wk2gt_4),479).
orientation(wk2gt_1,125,480). coords(wk2gt_1,115,53,480).  visible(wk2gt_1,480). happensAt(inactive(wk2gt_1),480). orientation(wk2gt_4,146,480). coords(wk2gt_4,75,97,480).  visible(wk2gt_4,480). happensAt(walking(wk2gt_4),480).
orientation(wk2gt_1,125,481). coords(wk2gt_1,115,53,481).  visible(wk2gt_1,481). happensAt(inactive(wk2gt_1),481). orientation(wk2gt_4,146,481). coords(wk2gt_4,74,97,481).  visible(wk2gt_4,481). happensAt(walking(wk2gt_4),481).
orientation(wk2gt_1,125,482). coords(wk2gt_1,115,53,482).  visible(wk2gt_1,482). happensAt(inactive(wk2gt_1),482). orientation(wk2gt_4,146,482). coords(wk2gt_4,74,97,482).  visible(wk2gt_4,482). happensAt(walking(wk2gt_4),482).
orientation(wk2gt_1,125,483). coords(wk2gt_1,115,53,483).  visible(wk2gt_1,483). happensAt(inactive(wk2gt_1),483). orientation(wk2gt_4,146,483). coords(wk2gt_4,74,97,483).  visible(wk2gt_4,483). happensAt(walking(wk2gt_4),483).
orientation(wk2gt_1,125,484). coords(wk2gt_1,115,53,484).  visible(wk2gt_1,484). happensAt(inactive(wk2gt_1),484). orientation(wk2gt_4,146,484). coords(wk2gt_4,74,97,484).  visible(wk2gt_4,484). happensAt(walking(wk2gt_4),484).
orientation(wk2gt_1,125,485). coords(wk2gt_1,115,53,485).  visible(wk2gt_1,485). happensAt(inactive(wk2gt_1),485). orientation(wk2gt_4,146,485). coords(wk2gt_4,73,97,485).  visible(wk2gt_4,485). happensAt(walking(wk2gt_4),485).
orientation(wk2gt_1,125,486). coords(wk2gt_1,115,53,486).  visible(wk2gt_1,486). happensAt(inactive(wk2gt_1),486). orientation(wk2gt_4,146,486). coords(wk2gt_4,73,97,486).  visible(wk2gt_4,486). happensAt(walking(wk2gt_4),486).
orientation(wk2gt_1,125,487). coords(wk2gt_1,115,53,487).  visible(wk2gt_1,487). happensAt(inactive(wk2gt_1),487). orientation(wk2gt_4,146,487). coords(wk2gt_4,72,97,487).  visible(wk2gt_4,487). happensAt(walking(wk2gt_4),487).
orientation(wk2gt_1,125,488). coords(wk2gt_1,115,53,488).  visible(wk2gt_1,488). happensAt(inactive(wk2gt_1),488). orientation(wk2gt_4,150,488). coords(wk2gt_4,72,97,488).  visible(wk2gt_4,488). happensAt(walking(wk2gt_4),488).
orientation(wk2gt_1,125,489). coords(wk2gt_1,115,53,489).  visible(wk2gt_1,489). happensAt(inactive(wk2gt_1),489). orientation(wk2gt_4,150,489). coords(wk2gt_4,70,97,489).  visible(wk2gt_4,489). happensAt(walking(wk2gt_4),489).
orientation(wk2gt_1,122,490). coords(wk2gt_1,115,53,490).  visible(wk2gt_1,490). happensAt(inactive(wk2gt_1),490). orientation(wk2gt_4,150,490). coords(wk2gt_4,69,97,490).  visible(wk2gt_4,490). happensAt(walking(wk2gt_4),490).
orientation(wk2gt_1,122,491). coords(wk2gt_1,115,53,491).  visible(wk2gt_1,491). happensAt(inactive(wk2gt_1),491). orientation(wk2gt_4,150,491). coords(wk2gt_4,69,97,491).  visible(wk2gt_4,491). happensAt(walking(wk2gt_4),491).
orientation(wk2gt_1,122,492). coords(wk2gt_1,115,53,492).  visible(wk2gt_1,492). happensAt(inactive(wk2gt_1),492). orientation(wk2gt_4,150,492). coords(wk2gt_4,69,97,492).  visible(wk2gt_4,492). happensAt(walking(wk2gt_4),492).
orientation(wk2gt_1,122,493). coords(wk2gt_1,115,53,493).  visible(wk2gt_1,493). happensAt(inactive(wk2gt_1),493). orientation(wk2gt_4,149,493). coords(wk2gt_4,68,97,493).  visible(wk2gt_4,493). happensAt(walking(wk2gt_4),493).
orientation(wk2gt_1,122,494). coords(wk2gt_1,115,53,494).  visible(wk2gt_1,494). happensAt(inactive(wk2gt_1),494). orientation(wk2gt_4,149,494). coords(wk2gt_4,68,97,494).  visible(wk2gt_4,494). happensAt(walking(wk2gt_4),494).
orientation(wk2gt_1,122,495). coords(wk2gt_1,115,53,495).  visible(wk2gt_1,495). happensAt(inactive(wk2gt_1),495). orientation(wk2gt_4,149,495). coords(wk2gt_4,68,97,495).  visible(wk2gt_4,495). happensAt(walking(wk2gt_4),495).
orientation(wk2gt_1,122,496). coords(wk2gt_1,115,53,496).  visible(wk2gt_1,496). happensAt(inactive(wk2gt_1),496). orientation(wk2gt_4,146,496). coords(wk2gt_4,68,97,496).  visible(wk2gt_4,496). happensAt(walking(wk2gt_4),496).
orientation(wk2gt_1,122,497). coords(wk2gt_1,115,53,497).  visible(wk2gt_1,497). happensAt(inactive(wk2gt_1),497). orientation(wk2gt_4,146,497). coords(wk2gt_4,68,97,497).  visible(wk2gt_4,497). happensAt(walking(wk2gt_4),497).
orientation(wk2gt_1,122,498). coords(wk2gt_1,115,53,498).  visible(wk2gt_1,498). happensAt(inactive(wk2gt_1),498). orientation(wk2gt_4,146,498). coords(wk2gt_4,67,97,498).  visible(wk2gt_4,498). happensAt(walking(wk2gt_4),498).
orientation(wk2gt_1,122,499). coords(wk2gt_1,115,53,499).  visible(wk2gt_1,499). happensAt(inactive(wk2gt_1),499). orientation(wk2gt_4,146,499). coords(wk2gt_4,67,97,499).  visible(wk2gt_4,499). happensAt(walking(wk2gt_4),499).
#show holdsAt/2.