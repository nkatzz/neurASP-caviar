
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

	orientation(wk1gt_1,147,476). coords(wk1gt_1,81,79,476).  visible(wk1gt_1,476). happensAt(walking(wk1gt_1),476). orientation(wk1gt_5,162,476). coords(wk1gt_5,57,152,476).  visible(wk1gt_5,476). happensAt(active(wk1gt_5),476).
orientation(wk1gt_1,147,477). coords(wk1gt_1,81,78,477).  visible(wk1gt_1,477). happensAt(walking(wk1gt_1),477). orientation(wk1gt_5,162,477). coords(wk1gt_5,57,152,477).  visible(wk1gt_5,477). happensAt(active(wk1gt_5),477).
orientation(wk1gt_1,147,478). coords(wk1gt_1,81,78,478).  visible(wk1gt_1,478). happensAt(walking(wk1gt_1),478). orientation(wk1gt_5,162,478). coords(wk1gt_5,57,152,478).  visible(wk1gt_5,478). happensAt(active(wk1gt_5),478).
orientation(wk1gt_1,147,479). coords(wk1gt_1,81,78,479).  visible(wk1gt_1,479). happensAt(walking(wk1gt_1),479). orientation(wk1gt_5,162,479). coords(wk1gt_5,57,152,479).  visible(wk1gt_5,479). happensAt(active(wk1gt_5),479).
orientation(wk1gt_1,140,480). coords(wk1gt_1,81,78,480).  visible(wk1gt_1,480). happensAt(walking(wk1gt_1),480). orientation(wk1gt_5,162,480). coords(wk1gt_5,57,152,480).  visible(wk1gt_5,480). happensAt(active(wk1gt_5),480).
orientation(wk1gt_1,140,481). coords(wk1gt_1,81,78,481).  visible(wk1gt_1,481). happensAt(walking(wk1gt_1),481). orientation(wk1gt_5,162,481). coords(wk1gt_5,57,152,481).  visible(wk1gt_5,481). happensAt(active(wk1gt_5),481).
orientation(wk1gt_1,140,482). coords(wk1gt_1,81,77,482).  visible(wk1gt_1,482). happensAt(walking(wk1gt_1),482). orientation(wk1gt_5,162,482). coords(wk1gt_5,57,152,482).  visible(wk1gt_5,482). happensAt(active(wk1gt_5),482).
orientation(wk1gt_1,140,483). coords(wk1gt_1,81,76,483).  visible(wk1gt_1,483). happensAt(walking(wk1gt_1),483). orientation(wk1gt_5,162,483). coords(wk1gt_5,57,152,483).  visible(wk1gt_5,483). happensAt(active(wk1gt_5),483).
orientation(wk1gt_1,140,484). coords(wk1gt_1,81,75,484).  visible(wk1gt_1,484). happensAt(walking(wk1gt_1),484). orientation(wk1gt_5,162,484). coords(wk1gt_5,57,152,484).  visible(wk1gt_5,484). happensAt(active(wk1gt_5),484).
orientation(wk1gt_1,140,485). coords(wk1gt_1,81,75,485).  visible(wk1gt_1,485). happensAt(walking(wk1gt_1),485). orientation(wk1gt_5,162,485). coords(wk1gt_5,57,152,485).  visible(wk1gt_5,485). happensAt(active(wk1gt_5),485).
orientation(wk1gt_1,140,486). coords(wk1gt_1,81,73,486).  visible(wk1gt_1,486). happensAt(walking(wk1gt_1),486). orientation(wk1gt_5,162,486). coords(wk1gt_5,57,152,486).  visible(wk1gt_5,486). happensAt(active(wk1gt_5),486).
orientation(wk1gt_1,140,487). coords(wk1gt_1,81,72,487).  visible(wk1gt_1,487). happensAt(walking(wk1gt_1),487). orientation(wk1gt_5,162,487). coords(wk1gt_5,57,152,487).  visible(wk1gt_5,487). happensAt(active(wk1gt_5),487).
orientation(wk1gt_1,140,488). coords(wk1gt_1,81,72,488).  visible(wk1gt_1,488). happensAt(walking(wk1gt_1),488). orientation(wk1gt_5,162,488). coords(wk1gt_5,57,152,488).  visible(wk1gt_5,488). happensAt(active(wk1gt_5),488).
orientation(wk1gt_1,140,489). coords(wk1gt_1,81,72,489).  visible(wk1gt_1,489). happensAt(walking(wk1gt_1),489). orientation(wk1gt_5,162,489). coords(wk1gt_5,57,151,489).  visible(wk1gt_5,489). happensAt(active(wk1gt_5),489).
orientation(wk1gt_1,140,490). coords(wk1gt_1,81,72,490).  visible(wk1gt_1,490). happensAt(walking(wk1gt_1),490). orientation(wk1gt_5,162,490). coords(wk1gt_5,57,150,490).  visible(wk1gt_5,490). happensAt(active(wk1gt_5),490).
orientation(wk1gt_1,140,491). coords(wk1gt_1,81,72,491).  visible(wk1gt_1,491). happensAt(walking(wk1gt_1),491). orientation(wk1gt_5,162,491). coords(wk1gt_5,57,150,491).  visible(wk1gt_5,491). happensAt(active(wk1gt_5),491).
orientation(wk1gt_1,140,492). coords(wk1gt_1,81,71,492).  visible(wk1gt_1,492). happensAt(walking(wk1gt_1),492). orientation(wk1gt_5,162,492). coords(wk1gt_5,57,150,492).  visible(wk1gt_5,492). happensAt(active(wk1gt_5),492).
orientation(wk1gt_1,140,493). coords(wk1gt_1,81,70,493).  visible(wk1gt_1,493). happensAt(walking(wk1gt_1),493). orientation(wk1gt_5,162,493). coords(wk1gt_5,57,150,493).  visible(wk1gt_5,493). happensAt(active(wk1gt_5),493).
orientation(wk1gt_1,140,494). coords(wk1gt_1,81,70,494).  visible(wk1gt_1,494). happensAt(walking(wk1gt_1),494). orientation(wk1gt_5,162,494). coords(wk1gt_5,57,150,494).  visible(wk1gt_5,494). happensAt(active(wk1gt_5),494).
orientation(wk1gt_1,140,495). coords(wk1gt_1,81,69,495).  visible(wk1gt_1,495). happensAt(walking(wk1gt_1),495). orientation(wk1gt_5,162,495). coords(wk1gt_5,57,150,495).  visible(wk1gt_5,495). happensAt(active(wk1gt_5),495).
orientation(wk1gt_1,140,496). coords(wk1gt_1,80,69,496).  visible(wk1gt_1,496). happensAt(walking(wk1gt_1),496). orientation(wk1gt_5,162,496). coords(wk1gt_5,57,150,496).  visible(wk1gt_5,496). happensAt(active(wk1gt_5),496).
orientation(wk1gt_1,140,497). coords(wk1gt_1,80,68,497).  visible(wk1gt_1,497). happensAt(walking(wk1gt_1),497). orientation(wk1gt_5,162,497). coords(wk1gt_5,57,150,497).  visible(wk1gt_5,497). happensAt(active(wk1gt_5),497).
orientation(wk1gt_1,140,498). coords(wk1gt_1,80,67,498).  visible(wk1gt_1,498). happensAt(walking(wk1gt_1),498). orientation(wk1gt_5,162,498). coords(wk1gt_5,57,150,498).  visible(wk1gt_5,498). happensAt(inactive(wk1gt_5),498).
orientation(wk1gt_1,140,499). coords(wk1gt_1,80,67,499).  visible(wk1gt_1,499). happensAt(walking(wk1gt_1),499). orientation(wk1gt_5,162,499). coords(wk1gt_5,57,150,499).  visible(wk1gt_5,499). happensAt(inactive(wk1gt_5),499).
orientation(wk1gt_1,140,500). coords(wk1gt_1,80,66,500).  visible(wk1gt_1,500). happensAt(walking(wk1gt_1),500). orientation(wk1gt_5,162,500). coords(wk1gt_5,57,150,500).  visible(wk1gt_5,500). happensAt(inactive(wk1gt_5),500).
#show holdsAt/2.