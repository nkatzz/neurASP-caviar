
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

	orientation(lb2gt_4,154,479). coords(lb2gt_4,68,118,479).  visible(lb2gt_4,479). happensAt(walking(lb2gt_4),479). orientation(lb2gt_6,151,479). coords(lb2gt_6,72,122,479).  visible(lb2gt_6,479). happensAt(inactive(lb2gt_6),479).
orientation(lb2gt_4,154,480). coords(lb2gt_4,68,118,480).  visible(lb2gt_4,480). happensAt(walking(lb2gt_4),480). orientation(lb2gt_6,151,480). coords(lb2gt_6,72,122,480).  visible(lb2gt_6,480). happensAt(inactive(lb2gt_6),480).
orientation(lb2gt_4,154,481). coords(lb2gt_4,68,116,481).  visible(lb2gt_4,481). happensAt(walking(lb2gt_4),481). orientation(lb2gt_6,151,481). coords(lb2gt_6,72,122,481).  visible(lb2gt_6,481). happensAt(inactive(lb2gt_6),481).
orientation(lb2gt_4,154,482). coords(lb2gt_4,68,115,482).  visible(lb2gt_4,482). happensAt(walking(lb2gt_4),482). orientation(lb2gt_6,151,482). coords(lb2gt_6,72,122,482).  visible(lb2gt_6,482). happensAt(inactive(lb2gt_6),482).
orientation(lb2gt_4,154,483). coords(lb2gt_4,68,115,483).  visible(lb2gt_4,483). happensAt(walking(lb2gt_4),483). orientation(lb2gt_6,151,483). coords(lb2gt_6,72,122,483).  visible(lb2gt_6,483). happensAt(inactive(lb2gt_6),483).
orientation(lb2gt_4,154,484). coords(lb2gt_4,68,115,484).  visible(lb2gt_4,484). happensAt(walking(lb2gt_4),484). orientation(lb2gt_6,151,484). coords(lb2gt_6,72,122,484).  visible(lb2gt_6,484). happensAt(inactive(lb2gt_6),484).
orientation(lb2gt_4,154,485). coords(lb2gt_4,68,114,485).  visible(lb2gt_4,485). happensAt(walking(lb2gt_4),485). orientation(lb2gt_6,151,485). coords(lb2gt_6,72,122,485).  visible(lb2gt_6,485). happensAt(inactive(lb2gt_6),485).
orientation(lb2gt_4,154,486). coords(lb2gt_4,69,114,486).  visible(lb2gt_4,486). happensAt(walking(lb2gt_4),486). orientation(lb2gt_6,151,486). coords(lb2gt_6,72,122,486).  visible(lb2gt_6,486). happensAt(inactive(lb2gt_6),486).
orientation(lb2gt_4,154,487). coords(lb2gt_4,69,113,487).  visible(lb2gt_4,487). happensAt(walking(lb2gt_4),487). orientation(lb2gt_6,151,487). coords(lb2gt_6,72,122,487).  visible(lb2gt_6,487). happensAt(inactive(lb2gt_6),487).
orientation(lb2gt_4,154,488). coords(lb2gt_4,70,113,488).  visible(lb2gt_4,488). happensAt(walking(lb2gt_4),488). orientation(lb2gt_6,151,488). coords(lb2gt_6,72,122,488).  visible(lb2gt_6,488). happensAt(inactive(lb2gt_6),488).
orientation(lb2gt_4,154,489). coords(lb2gt_4,71,113,489).  visible(lb2gt_4,489). happensAt(walking(lb2gt_4),489). orientation(lb2gt_6,151,489). coords(lb2gt_6,72,122,489).  visible(lb2gt_6,489). happensAt(inactive(lb2gt_6),489).
orientation(lb2gt_4,154,490). coords(lb2gt_4,71,112,490).  visible(lb2gt_4,490). happensAt(walking(lb2gt_4),490). orientation(lb2gt_6,151,490). coords(lb2gt_6,72,122,490).  visible(lb2gt_6,490). happensAt(inactive(lb2gt_6),490).
orientation(lb2gt_4,154,491). coords(lb2gt_4,72,111,491).  visible(lb2gt_4,491). happensAt(walking(lb2gt_4),491). orientation(lb2gt_6,151,491). coords(lb2gt_6,72,122,491).  visible(lb2gt_6,491). happensAt(inactive(lb2gt_6),491).
orientation(lb2gt_4,154,492). coords(lb2gt_4,73,111,492).  visible(lb2gt_4,492). happensAt(walking(lb2gt_4),492). orientation(lb2gt_6,151,492). coords(lb2gt_6,72,122,492).  visible(lb2gt_6,492). happensAt(inactive(lb2gt_6),492).
orientation(lb2gt_4,154,493). coords(lb2gt_4,73,110,493).  visible(lb2gt_4,493). happensAt(walking(lb2gt_4),493). orientation(lb2gt_6,151,493). coords(lb2gt_6,72,122,493).  visible(lb2gt_6,493). happensAt(inactive(lb2gt_6),493).
orientation(lb2gt_4,154,494). coords(lb2gt_4,73,109,494).  visible(lb2gt_4,494). happensAt(walking(lb2gt_4),494). orientation(lb2gt_6,151,494). coords(lb2gt_6,72,122,494).  visible(lb2gt_6,494). happensAt(inactive(lb2gt_6),494).
orientation(lb2gt_4,154,495). coords(lb2gt_4,73,107,495).  visible(lb2gt_4,495). happensAt(walking(lb2gt_4),495). orientation(lb2gt_6,151,495). coords(lb2gt_6,72,122,495).  visible(lb2gt_6,495). happensAt(inactive(lb2gt_6),495).
orientation(lb2gt_4,154,496). coords(lb2gt_4,73,106,496).  visible(lb2gt_4,496). happensAt(walking(lb2gt_4),496). orientation(lb2gt_6,151,496). coords(lb2gt_6,72,122,496).  visible(lb2gt_6,496). happensAt(inactive(lb2gt_6),496).
orientation(lb2gt_4,154,497). coords(lb2gt_4,73,104,497).  visible(lb2gt_4,497). happensAt(walking(lb2gt_4),497). orientation(lb2gt_6,151,497). coords(lb2gt_6,72,122,497).  visible(lb2gt_6,497). happensAt(inactive(lb2gt_6),497).
orientation(lb2gt_4,154,498). coords(lb2gt_4,73,102,498).  visible(lb2gt_4,498). happensAt(walking(lb2gt_4),498). orientation(lb2gt_6,151,498). coords(lb2gt_6,72,122,498).  visible(lb2gt_6,498). happensAt(inactive(lb2gt_6),498).
orientation(lb2gt_4,154,499). coords(lb2gt_4,74,101,499).  visible(lb2gt_4,499). happensAt(walking(lb2gt_4),499). orientation(lb2gt_6,151,499). coords(lb2gt_6,72,122,499).  visible(lb2gt_6,499). happensAt(inactive(lb2gt_6),499).
orientation(lb2gt_4,154,500). coords(lb2gt_4,75,101,500).  visible(lb2gt_4,500). happensAt(walking(lb2gt_4),500). orientation(lb2gt_6,151,500). coords(lb2gt_6,72,122,500).  visible(lb2gt_6,500). happensAt(inactive(lb2gt_6),500).
orientation(lb2gt_4,154,501). coords(lb2gt_4,76,100,501).  visible(lb2gt_4,501). happensAt(walking(lb2gt_4),501). orientation(lb2gt_6,151,501). coords(lb2gt_6,72,122,501).  visible(lb2gt_6,501). happensAt(inactive(lb2gt_6),501).
orientation(lb2gt_4,154,502). coords(lb2gt_4,78,100,502).  visible(lb2gt_4,502). happensAt(walking(lb2gt_4),502). orientation(lb2gt_6,151,502). coords(lb2gt_6,72,122,502).  visible(lb2gt_6,502). happensAt(inactive(lb2gt_6),502).
orientation(lb2gt_4,154,503). coords(lb2gt_4,79,99,503).  visible(lb2gt_4,503). happensAt(walking(lb2gt_4),503). orientation(lb2gt_6,151,503). coords(lb2gt_6,72,122,503).  visible(lb2gt_6,503). happensAt(inactive(lb2gt_6),503).
#show holdsAt/2.