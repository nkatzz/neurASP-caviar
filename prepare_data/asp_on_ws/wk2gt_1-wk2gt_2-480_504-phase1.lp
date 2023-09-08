
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

	orientation(wk2gt_1,125,480). coords(wk2gt_1,115,53,480).  visible(wk2gt_1,480). happensAt(inactive(wk2gt_1),480). orientation(wk2gt_2,141,480). coords(wk2gt_2,98,63,480).  visible(wk2gt_2,480). happensAt(inactive(wk2gt_2),480).
orientation(wk2gt_1,125,481). coords(wk2gt_1,115,53,481).  visible(wk2gt_1,481). happensAt(inactive(wk2gt_1),481). orientation(wk2gt_2,141,481). coords(wk2gt_2,98,63,481).  visible(wk2gt_2,481). happensAt(inactive(wk2gt_2),481).
orientation(wk2gt_1,125,482). coords(wk2gt_1,115,53,482).  visible(wk2gt_1,482). happensAt(inactive(wk2gt_1),482). orientation(wk2gt_2,141,482). coords(wk2gt_2,98,63,482).  visible(wk2gt_2,482). happensAt(inactive(wk2gt_2),482).
orientation(wk2gt_1,125,483). coords(wk2gt_1,115,53,483).  visible(wk2gt_1,483). happensAt(inactive(wk2gt_1),483). orientation(wk2gt_2,141,483). coords(wk2gt_2,98,63,483).  visible(wk2gt_2,483). happensAt(inactive(wk2gt_2),483).
orientation(wk2gt_1,125,484). coords(wk2gt_1,115,53,484).  visible(wk2gt_1,484). happensAt(inactive(wk2gt_1),484). orientation(wk2gt_2,141,484). coords(wk2gt_2,98,63,484).  visible(wk2gt_2,484). happensAt(inactive(wk2gt_2),484).
orientation(wk2gt_1,125,485). coords(wk2gt_1,115,53,485).  visible(wk2gt_1,485). happensAt(inactive(wk2gt_1),485). orientation(wk2gt_2,141,485). coords(wk2gt_2,98,63,485).  visible(wk2gt_2,485). happensAt(inactive(wk2gt_2),485).
orientation(wk2gt_1,125,486). coords(wk2gt_1,115,53,486).  visible(wk2gt_1,486). happensAt(inactive(wk2gt_1),486). orientation(wk2gt_2,141,486). coords(wk2gt_2,98,63,486).  visible(wk2gt_2,486). happensAt(inactive(wk2gt_2),486).
orientation(wk2gt_1,125,487). coords(wk2gt_1,115,53,487).  visible(wk2gt_1,487). happensAt(inactive(wk2gt_1),487). orientation(wk2gt_2,141,487). coords(wk2gt_2,98,63,487).  visible(wk2gt_2,487). happensAt(inactive(wk2gt_2),487).
orientation(wk2gt_1,125,488). coords(wk2gt_1,115,53,488).  visible(wk2gt_1,488). happensAt(inactive(wk2gt_1),488). orientation(wk2gt_2,141,488). coords(wk2gt_2,98,63,488).  visible(wk2gt_2,488). happensAt(inactive(wk2gt_2),488).
orientation(wk2gt_1,125,489). coords(wk2gt_1,115,53,489).  visible(wk2gt_1,489). happensAt(inactive(wk2gt_1),489). orientation(wk2gt_2,141,489). coords(wk2gt_2,99,63,489).  visible(wk2gt_2,489). happensAt(inactive(wk2gt_2),489).
orientation(wk2gt_1,122,490). coords(wk2gt_1,115,53,490).  visible(wk2gt_1,490). happensAt(inactive(wk2gt_1),490). orientation(wk2gt_2,141,490). coords(wk2gt_2,98,63,490).  visible(wk2gt_2,490). happensAt(inactive(wk2gt_2),490).
orientation(wk2gt_1,122,491). coords(wk2gt_1,115,53,491).  visible(wk2gt_1,491). happensAt(inactive(wk2gt_1),491). orientation(wk2gt_2,141,491). coords(wk2gt_2,99,63,491).  visible(wk2gt_2,491). happensAt(inactive(wk2gt_2),491).
orientation(wk2gt_1,122,492). coords(wk2gt_1,115,53,492).  visible(wk2gt_1,492). happensAt(inactive(wk2gt_1),492). orientation(wk2gt_2,141,492). coords(wk2gt_2,99,63,492).  visible(wk2gt_2,492). happensAt(inactive(wk2gt_2),492).
orientation(wk2gt_1,122,493). coords(wk2gt_1,115,53,493).  visible(wk2gt_1,493). happensAt(inactive(wk2gt_1),493). orientation(wk2gt_2,141,493). coords(wk2gt_2,99,63,493).  visible(wk2gt_2,493). happensAt(inactive(wk2gt_2),493).
orientation(wk2gt_1,122,494). coords(wk2gt_1,115,53,494).  visible(wk2gt_1,494). happensAt(inactive(wk2gt_1),494). orientation(wk2gt_2,141,494). coords(wk2gt_2,99,63,494).  visible(wk2gt_2,494). happensAt(inactive(wk2gt_2),494).
orientation(wk2gt_1,122,495). coords(wk2gt_1,115,53,495).  visible(wk2gt_1,495). happensAt(inactive(wk2gt_1),495). orientation(wk2gt_2,141,495). coords(wk2gt_2,100,63,495).  visible(wk2gt_2,495). happensAt(inactive(wk2gt_2),495).
orientation(wk2gt_1,122,496). coords(wk2gt_1,115,53,496).  visible(wk2gt_1,496). happensAt(inactive(wk2gt_1),496). orientation(wk2gt_2,141,496). coords(wk2gt_2,99,63,496).  visible(wk2gt_2,496). happensAt(inactive(wk2gt_2),496).
orientation(wk2gt_1,122,497). coords(wk2gt_1,115,53,497).  visible(wk2gt_1,497). happensAt(inactive(wk2gt_1),497). orientation(wk2gt_2,141,497). coords(wk2gt_2,99,63,497).  visible(wk2gt_2,497). happensAt(inactive(wk2gt_2),497).
orientation(wk2gt_1,122,498). coords(wk2gt_1,115,53,498).  visible(wk2gt_1,498). happensAt(inactive(wk2gt_1),498). orientation(wk2gt_2,141,498). coords(wk2gt_2,99,63,498).  visible(wk2gt_2,498). happensAt(inactive(wk2gt_2),498).
orientation(wk2gt_1,122,499). coords(wk2gt_1,115,53,499).  visible(wk2gt_1,499). happensAt(inactive(wk2gt_1),499). orientation(wk2gt_2,141,499). coords(wk2gt_2,99,63,499).  visible(wk2gt_2,499). happensAt(inactive(wk2gt_2),499).
orientation(wk2gt_1,122,500). coords(wk2gt_1,115,53,500).  visible(wk2gt_1,500). happensAt(inactive(wk2gt_1),500). orientation(wk2gt_2,141,500). coords(wk2gt_2,99,63,500).  visible(wk2gt_2,500). happensAt(inactive(wk2gt_2),500).
orientation(wk2gt_1,122,501). coords(wk2gt_1,115,53,501).  visible(wk2gt_1,501). happensAt(inactive(wk2gt_1),501). orientation(wk2gt_2,141,501). coords(wk2gt_2,99,63,501).  visible(wk2gt_2,501). happensAt(inactive(wk2gt_2),501).
orientation(wk2gt_1,122,502). coords(wk2gt_1,115,53,502).  visible(wk2gt_1,502). happensAt(inactive(wk2gt_1),502). orientation(wk2gt_2,141,502). coords(wk2gt_2,99,63,502).  visible(wk2gt_2,502). happensAt(inactive(wk2gt_2),502).
orientation(wk2gt_1,122,503). coords(wk2gt_1,115,53,503).  visible(wk2gt_1,503). happensAt(inactive(wk2gt_1),503). orientation(wk2gt_2,141,503). coords(wk2gt_2,99,63,503).  visible(wk2gt_2,503). happensAt(inactive(wk2gt_2),503).
orientation(wk2gt_1,122,504). coords(wk2gt_1,115,53,504).  visible(wk2gt_1,504). happensAt(inactive(wk2gt_1),504). orientation(wk2gt_2,141,504). coords(wk2gt_2,99,63,504).  visible(wk2gt_2,504). happensAt(inactive(wk2gt_2),504).
#show holdsAt/2.