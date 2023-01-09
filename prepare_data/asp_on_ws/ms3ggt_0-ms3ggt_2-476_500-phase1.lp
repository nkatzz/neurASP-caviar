
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

	orientation(ms3ggt_0,151,476). coords(ms3ggt_0,79,172,476).  visible(ms3ggt_0,476). happensAt(walking(ms3ggt_0),476). orientation(ms3ggt_2,158,476). coords(ms3ggt_2,99,141,476).  visible(ms3ggt_2,476). happensAt(walking(ms3ggt_2),476).
orientation(ms3ggt_0,151,477). coords(ms3ggt_0,79,172,477).  visible(ms3ggt_0,477). happensAt(walking(ms3ggt_0),477). orientation(ms3ggt_2,158,477). coords(ms3ggt_2,98,141,477).  visible(ms3ggt_2,477). happensAt(walking(ms3ggt_2),477).
orientation(ms3ggt_0,151,478). coords(ms3ggt_0,79,172,478).  visible(ms3ggt_0,478). happensAt(walking(ms3ggt_0),478). orientation(ms3ggt_2,158,478). coords(ms3ggt_2,98,141,478).  visible(ms3ggt_2,478). happensAt(walking(ms3ggt_2),478).
orientation(ms3ggt_0,151,479). coords(ms3ggt_0,79,172,479).  visible(ms3ggt_0,479). happensAt(walking(ms3ggt_0),479). orientation(ms3ggt_2,158,479). coords(ms3ggt_2,98,139,479).  visible(ms3ggt_2,479). happensAt(walking(ms3ggt_2),479).
orientation(ms3ggt_0,151,480). coords(ms3ggt_0,79,172,480).  visible(ms3ggt_0,480). happensAt(walking(ms3ggt_0),480). orientation(ms3ggt_2,158,480). coords(ms3ggt_2,97,137,480).  visible(ms3ggt_2,480). happensAt(walking(ms3ggt_2),480).
orientation(ms3ggt_0,156,481). coords(ms3ggt_0,78,178,481).  visible(ms3ggt_0,481). happensAt(walking(ms3ggt_0),481). orientation(ms3ggt_2,158,481). coords(ms3ggt_2,97,136,481).  visible(ms3ggt_2,481). happensAt(walking(ms3ggt_2),481).
orientation(ms3ggt_0,156,482). coords(ms3ggt_0,78,179,482).  visible(ms3ggt_0,482). happensAt(walking(ms3ggt_0),482). orientation(ms3ggt_2,158,482). coords(ms3ggt_2,96,135,482).  visible(ms3ggt_2,482). happensAt(walking(ms3ggt_2),482).
orientation(ms3ggt_0,156,483). coords(ms3ggt_0,78,181,483).  visible(ms3ggt_0,483). happensAt(walking(ms3ggt_0),483). orientation(ms3ggt_2,158,483). coords(ms3ggt_2,95,135,483).  visible(ms3ggt_2,483). happensAt(walking(ms3ggt_2),483).
orientation(ms3ggt_0,156,484). coords(ms3ggt_0,78,181,484).  visible(ms3ggt_0,484). happensAt(walking(ms3ggt_0),484). orientation(ms3ggt_2,158,484). coords(ms3ggt_2,95,134,484).  visible(ms3ggt_2,484). happensAt(walking(ms3ggt_2),484).
orientation(ms3ggt_0,156,485). coords(ms3ggt_0,78,182,485).  visible(ms3ggt_0,485). happensAt(walking(ms3ggt_0),485). orientation(ms3ggt_2,158,485). coords(ms3ggt_2,95,134,485).  visible(ms3ggt_2,485). happensAt(walking(ms3ggt_2),485).
orientation(ms3ggt_0,156,486). coords(ms3ggt_0,77,182,486).  visible(ms3ggt_0,486). happensAt(walking(ms3ggt_0),486). orientation(ms3ggt_2,158,486). coords(ms3ggt_2,95,133,486).  visible(ms3ggt_2,486). happensAt(walking(ms3ggt_2),486).
orientation(ms3ggt_0,156,487). coords(ms3ggt_0,77,182,487).  visible(ms3ggt_0,487). happensAt(walking(ms3ggt_0),487). orientation(ms3ggt_2,150,487). coords(ms3ggt_2,95,133,487).  visible(ms3ggt_2,487). happensAt(walking(ms3ggt_2),487).
orientation(ms3ggt_0,160,488). coords(ms3ggt_0,76,183,488).  visible(ms3ggt_0,488). happensAt(walking(ms3ggt_0),488). orientation(ms3ggt_2,150,488). coords(ms3ggt_2,94,133,488).  visible(ms3ggt_2,488). happensAt(walking(ms3ggt_2),488).
orientation(ms3ggt_0,160,489). coords(ms3ggt_0,76,184,489).  visible(ms3ggt_0,489). happensAt(walking(ms3ggt_0),489). orientation(ms3ggt_2,150,489). coords(ms3ggt_2,94,132,489).  visible(ms3ggt_2,489). happensAt(walking(ms3ggt_2),489).
orientation(ms3ggt_0,160,490). coords(ms3ggt_0,76,184,490).  visible(ms3ggt_0,490). happensAt(walking(ms3ggt_0),490). orientation(ms3ggt_2,150,490). coords(ms3ggt_2,94,132,490).  visible(ms3ggt_2,490). happensAt(walking(ms3ggt_2),490).
orientation(ms3ggt_0,163,491). coords(ms3ggt_0,75,185,491).  visible(ms3ggt_0,491). happensAt(walking(ms3ggt_0),491). orientation(ms3ggt_2,150,491). coords(ms3ggt_2,94,131,491).  visible(ms3ggt_2,491). happensAt(walking(ms3ggt_2),491).
orientation(ms3ggt_0,163,492). coords(ms3ggt_0,75,185,492).  visible(ms3ggt_0,492). happensAt(walking(ms3ggt_0),492). orientation(ms3ggt_2,143,492). coords(ms3ggt_2,94,131,492).  visible(ms3ggt_2,492). happensAt(walking(ms3ggt_2),492).
orientation(ms3ggt_0,163,493). coords(ms3ggt_0,75,186,493).  visible(ms3ggt_0,493). happensAt(walking(ms3ggt_0),493). orientation(ms3ggt_2,143,493). coords(ms3ggt_2,94,129,493).  visible(ms3ggt_2,493). happensAt(walking(ms3ggt_2),493).
orientation(ms3ggt_0,163,494). coords(ms3ggt_0,75,186,494).  visible(ms3ggt_0,494). happensAt(walking(ms3ggt_0),494). orientation(ms3ggt_2,143,494). coords(ms3ggt_2,93,127,494).  visible(ms3ggt_2,494). happensAt(walking(ms3ggt_2),494).
orientation(ms3ggt_0,163,495). coords(ms3ggt_0,75,187,495).  visible(ms3ggt_0,495). happensAt(walking(ms3ggt_0),495). orientation(ms3ggt_2,143,495). coords(ms3ggt_2,93,126,495).  visible(ms3ggt_2,495). happensAt(walking(ms3ggt_2),495).
orientation(ms3ggt_0,163,496). coords(ms3ggt_0,75,188,496).  visible(ms3ggt_0,496). happensAt(walking(ms3ggt_0),496). orientation(ms3ggt_2,143,496). coords(ms3ggt_2,93,125,496).  visible(ms3ggt_2,496). happensAt(walking(ms3ggt_2),496).
orientation(ms3ggt_0,163,497). coords(ms3ggt_0,75,188,497).  visible(ms3ggt_0,497). happensAt(walking(ms3ggt_0),497). orientation(ms3ggt_2,143,497). coords(ms3ggt_2,93,124,497).  visible(ms3ggt_2,497). happensAt(walking(ms3ggt_2),497).
orientation(ms3ggt_0,163,498). coords(ms3ggt_0,75,189,498).  visible(ms3ggt_0,498). happensAt(walking(ms3ggt_0),498). orientation(ms3ggt_2,143,498). coords(ms3ggt_2,93,123,498).  visible(ms3ggt_2,498). happensAt(walking(ms3ggt_2),498).
orientation(ms3ggt_0,163,499). coords(ms3ggt_0,75,189,499).  visible(ms3ggt_0,499). happensAt(walking(ms3ggt_0),499). orientation(ms3ggt_2,143,499). coords(ms3ggt_2,93,122,499).  visible(ms3ggt_2,499). happensAt(walking(ms3ggt_2),499).
orientation(ms3ggt_0,163,500). coords(ms3ggt_0,75,189,500).  visible(ms3ggt_0,500). happensAt(walking(ms3ggt_0),500). orientation(ms3ggt_2,147,500). coords(ms3ggt_2,93,120,500).  visible(ms3ggt_2,500). happensAt(walking(ms3ggt_2),500).
#show holdsAt/2.