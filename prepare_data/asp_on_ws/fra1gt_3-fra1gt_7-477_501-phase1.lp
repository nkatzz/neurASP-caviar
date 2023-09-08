
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

	orientation(fra1gt_3,175,477). coords(fra1gt_3,102,205,477).  visible(fra1gt_3,477). happensAt(active(fra1gt_3),477). orientation(fra1gt_7,146,477). coords(fra1gt_7,49,90,477).  visible(fra1gt_7,477). happensAt(walking(fra1gt_7),477).
orientation(fra1gt_3,175,478). coords(fra1gt_3,104,205,478).  visible(fra1gt_3,478). happensAt(active(fra1gt_3),478). orientation(fra1gt_7,146,478). coords(fra1gt_7,48,90,478).  visible(fra1gt_7,478). happensAt(walking(fra1gt_7),478).
orientation(fra1gt_3,175,479). coords(fra1gt_3,104,205,479).  visible(fra1gt_3,479). happensAt(active(fra1gt_3),479). orientation(fra1gt_7,146,479). coords(fra1gt_7,48,90,479).  visible(fra1gt_7,479). happensAt(walking(fra1gt_7),479).
orientation(fra1gt_3,175,480). coords(fra1gt_3,104,205,480).  visible(fra1gt_3,480). happensAt(active(fra1gt_3),480). orientation(fra1gt_7,146,480). coords(fra1gt_7,48,90,480).  visible(fra1gt_7,480). happensAt(walking(fra1gt_7),480).
orientation(fra1gt_3,175,481). coords(fra1gt_3,104,204,481).  visible(fra1gt_3,481). happensAt(active(fra1gt_3),481). orientation(fra1gt_7,146,481). coords(fra1gt_7,48,90,481).  visible(fra1gt_7,481). happensAt(walking(fra1gt_7),481).
orientation(fra1gt_3,175,482). coords(fra1gt_3,104,204,482).  visible(fra1gt_3,482). happensAt(active(fra1gt_3),482). orientation(fra1gt_7,146,482). coords(fra1gt_7,48,91,482).  visible(fra1gt_7,482). happensAt(walking(fra1gt_7),482).
orientation(fra1gt_3,175,483). coords(fra1gt_3,104,204,483).  visible(fra1gt_3,483). happensAt(active(fra1gt_3),483). orientation(fra1gt_7,146,483). coords(fra1gt_7,48,91,483).  visible(fra1gt_7,483). happensAt(walking(fra1gt_7),483).
orientation(fra1gt_3,175,484). coords(fra1gt_3,104,204,484).  visible(fra1gt_3,484). happensAt(active(fra1gt_3),484). orientation(fra1gt_7,146,484). coords(fra1gt_7,48,91,484).  visible(fra1gt_7,484). happensAt(walking(fra1gt_7),484).
orientation(fra1gt_3,175,485). coords(fra1gt_3,104,204,485).  visible(fra1gt_3,485). happensAt(active(fra1gt_3),485). orientation(fra1gt_7,146,485). coords(fra1gt_7,48,92,485).  visible(fra1gt_7,485). happensAt(walking(fra1gt_7),485).
orientation(fra1gt_3,175,486). coords(fra1gt_3,104,204,486).  visible(fra1gt_3,486). happensAt(active(fra1gt_3),486). orientation(fra1gt_7,146,486). coords(fra1gt_7,48,92,486).  visible(fra1gt_7,486). happensAt(walking(fra1gt_7),486).
orientation(fra1gt_3,175,487). coords(fra1gt_3,104,204,487).  visible(fra1gt_3,487). happensAt(active(fra1gt_3),487). orientation(fra1gt_7,146,487). coords(fra1gt_7,48,92,487).  visible(fra1gt_7,487). happensAt(walking(fra1gt_7),487).
orientation(fra1gt_3,175,488). coords(fra1gt_3,101,204,488).  visible(fra1gt_3,488). happensAt(active(fra1gt_3),488). orientation(fra1gt_7,146,488). coords(fra1gt_7,47,92,488).  visible(fra1gt_7,488). happensAt(walking(fra1gt_7),488).
orientation(fra1gt_3,175,489). coords(fra1gt_3,101,204,489).  visible(fra1gt_3,489). happensAt(active(fra1gt_3),489). orientation(fra1gt_7,146,489). coords(fra1gt_7,47,92,489).  visible(fra1gt_7,489). happensAt(walking(fra1gt_7),489).
orientation(fra1gt_3,175,490). coords(fra1gt_3,101,204,490).  visible(fra1gt_3,490). happensAt(active(fra1gt_3),490). orientation(fra1gt_7,146,490). coords(fra1gt_7,47,94,490).  visible(fra1gt_7,490). happensAt(walking(fra1gt_7),490).
orientation(fra1gt_3,175,491). coords(fra1gt_3,101,204,491).  visible(fra1gt_3,491). happensAt(active(fra1gt_3),491). orientation(fra1gt_7,146,491). coords(fra1gt_7,47,94,491).  visible(fra1gt_7,491). happensAt(walking(fra1gt_7),491).
orientation(fra1gt_3,175,492). coords(fra1gt_3,101,204,492).  visible(fra1gt_3,492). happensAt(active(fra1gt_3),492). orientation(fra1gt_7,146,492). coords(fra1gt_7,47,95,492).  visible(fra1gt_7,492). happensAt(walking(fra1gt_7),492).
orientation(fra1gt_3,175,493). coords(fra1gt_3,101,204,493).  visible(fra1gt_3,493). happensAt(active(fra1gt_3),493). orientation(fra1gt_7,146,493). coords(fra1gt_7,47,95,493).  visible(fra1gt_7,493). happensAt(walking(fra1gt_7),493).
orientation(fra1gt_3,175,494). coords(fra1gt_3,101,204,494).  visible(fra1gt_3,494). happensAt(active(fra1gt_3),494). orientation(fra1gt_7,146,494). coords(fra1gt_7,47,95,494).  visible(fra1gt_7,494). happensAt(walking(fra1gt_7),494).
orientation(fra1gt_3,175,495). coords(fra1gt_3,101,204,495).  visible(fra1gt_3,495). happensAt(active(fra1gt_3),495). orientation(fra1gt_7,146,495). coords(fra1gt_7,47,95,495).  visible(fra1gt_7,495). happensAt(walking(fra1gt_7),495).
orientation(fra1gt_3,175,496). coords(fra1gt_3,101,204,496).  visible(fra1gt_3,496). happensAt(active(fra1gt_3),496). orientation(fra1gt_7,146,496). coords(fra1gt_7,47,95,496).  visible(fra1gt_7,496). happensAt(walking(fra1gt_7),496).
orientation(fra1gt_3,175,497). coords(fra1gt_3,101,204,497).  visible(fra1gt_3,497). happensAt(active(fra1gt_3),497). orientation(fra1gt_7,146,497). coords(fra1gt_7,45,95,497).  visible(fra1gt_7,497). happensAt(walking(fra1gt_7),497).
orientation(fra1gt_3,175,498). coords(fra1gt_3,101,204,498).  visible(fra1gt_3,498). happensAt(active(fra1gt_3),498). orientation(fra1gt_7,146,498). coords(fra1gt_7,45,95,498).  visible(fra1gt_7,498). happensAt(walking(fra1gt_7),498).
orientation(fra1gt_3,175,499). coords(fra1gt_3,101,204,499).  visible(fra1gt_3,499). happensAt(active(fra1gt_3),499). orientation(fra1gt_7,146,499). coords(fra1gt_7,45,96,499).  visible(fra1gt_7,499). happensAt(walking(fra1gt_7),499).
orientation(fra1gt_3,175,500). coords(fra1gt_3,101,204,500).  visible(fra1gt_3,500). happensAt(active(fra1gt_3),500). orientation(fra1gt_7,146,500). coords(fra1gt_7,45,96,500).  visible(fra1gt_7,500). happensAt(walking(fra1gt_7),500).
orientation(fra1gt_3,175,501). coords(fra1gt_3,101,204,501).  visible(fra1gt_3,501). happensAt(active(fra1gt_3),501). orientation(fra1gt_7,146,501). coords(fra1gt_7,45,97,501).  visible(fra1gt_7,501). happensAt(walking(fra1gt_7),501).
#show holdsAt/2.