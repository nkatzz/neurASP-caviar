
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

	orientation(fomdgt3_1,150,472). coords(fomdgt3_1,46,102,472).  visible(fomdgt3_1,472). happensAt(inactive(fomdgt3_1),472). orientation(fomdgt3_6,160,472). coords(fomdgt3_6,49,103,472).  visible(fomdgt3_6,472). happensAt(walking(fomdgt3_6),472).
orientation(fomdgt3_1,150,473). coords(fomdgt3_1,46,102,473).  visible(fomdgt3_1,473). happensAt(inactive(fomdgt3_1),473). orientation(fomdgt3_6,160,473). coords(fomdgt3_6,49,103,473).  visible(fomdgt3_6,473). happensAt(walking(fomdgt3_6),473).
orientation(fomdgt3_1,150,474). coords(fomdgt3_1,46,102,474).  visible(fomdgt3_1,474). happensAt(inactive(fomdgt3_1),474). orientation(fomdgt3_6,160,474). coords(fomdgt3_6,49,103,474).  visible(fomdgt3_6,474). happensAt(walking(fomdgt3_6),474).
orientation(fomdgt3_1,150,475). coords(fomdgt3_1,46,102,475).  visible(fomdgt3_1,475). happensAt(inactive(fomdgt3_1),475). orientation(fomdgt3_6,160,475). coords(fomdgt3_6,49,103,475).  visible(fomdgt3_6,475). happensAt(walking(fomdgt3_6),475).
orientation(fomdgt3_1,150,476). coords(fomdgt3_1,46,102,476).  visible(fomdgt3_1,476). happensAt(inactive(fomdgt3_1),476). orientation(fomdgt3_6,160,476). coords(fomdgt3_6,49,103,476).  visible(fomdgt3_6,476). happensAt(walking(fomdgt3_6),476).
orientation(fomdgt3_1,150,477). coords(fomdgt3_1,46,102,477).  visible(fomdgt3_1,477). happensAt(inactive(fomdgt3_1),477). orientation(fomdgt3_6,160,477). coords(fomdgt3_6,49,103,477).  visible(fomdgt3_6,477). happensAt(walking(fomdgt3_6),477).
orientation(fomdgt3_1,150,478). coords(fomdgt3_1,46,102,478).  visible(fomdgt3_1,478). happensAt(inactive(fomdgt3_1),478). orientation(fomdgt3_6,160,478). coords(fomdgt3_6,49,103,478).  visible(fomdgt3_6,478). happensAt(walking(fomdgt3_6),478).
orientation(fomdgt3_1,150,479). coords(fomdgt3_1,46,102,479).  visible(fomdgt3_1,479). happensAt(inactive(fomdgt3_1),479). orientation(fomdgt3_6,160,479). coords(fomdgt3_6,49,103,479).  visible(fomdgt3_6,479). happensAt(walking(fomdgt3_6),479).
orientation(fomdgt3_1,150,480). coords(fomdgt3_1,46,102,480).  visible(fomdgt3_1,480). happensAt(inactive(fomdgt3_1),480). orientation(fomdgt3_6,160,480). coords(fomdgt3_6,49,103,480).  visible(fomdgt3_6,480). happensAt(walking(fomdgt3_6),480).
orientation(fomdgt3_1,150,481). coords(fomdgt3_1,46,102,481).  visible(fomdgt3_1,481). happensAt(inactive(fomdgt3_1),481). orientation(fomdgt3_6,160,481). coords(fomdgt3_6,49,103,481).  visible(fomdgt3_6,481). happensAt(walking(fomdgt3_6),481).
orientation(fomdgt3_1,150,482). coords(fomdgt3_1,46,102,482).  visible(fomdgt3_1,482). happensAt(inactive(fomdgt3_1),482). orientation(fomdgt3_6,160,482). coords(fomdgt3_6,49,103,482).  visible(fomdgt3_6,482). happensAt(walking(fomdgt3_6),482).
orientation(fomdgt3_1,150,483). coords(fomdgt3_1,46,102,483).  visible(fomdgt3_1,483). happensAt(inactive(fomdgt3_1),483). orientation(fomdgt3_6,160,483). coords(fomdgt3_6,49,103,483).  visible(fomdgt3_6,483). happensAt(walking(fomdgt3_6),483).
orientation(fomdgt3_1,150,484). coords(fomdgt3_1,46,102,484).  visible(fomdgt3_1,484). happensAt(inactive(fomdgt3_1),484). orientation(fomdgt3_6,177,484). coords(fomdgt3_6,42,108,484).  visible(fomdgt3_6,484). happensAt(walking(fomdgt3_6),484).
orientation(fomdgt3_1,150,485). coords(fomdgt3_1,46,102,485).  visible(fomdgt3_1,485). happensAt(inactive(fomdgt3_1),485). orientation(fomdgt3_6,177,485). coords(fomdgt3_6,42,108,485).  visible(fomdgt3_6,485). happensAt(walking(fomdgt3_6),485).
orientation(fomdgt3_1,150,486). coords(fomdgt3_1,46,102,486).  visible(fomdgt3_1,486). happensAt(inactive(fomdgt3_1),486). orientation(fomdgt3_6,177,486). coords(fomdgt3_6,42,111,486).  visible(fomdgt3_6,486). happensAt(walking(fomdgt3_6),486).
orientation(fomdgt3_1,150,487). coords(fomdgt3_1,46,102,487).  visible(fomdgt3_1,487). happensAt(inactive(fomdgt3_1),487). orientation(fomdgt3_6,177,487). coords(fomdgt3_6,42,111,487).  visible(fomdgt3_6,487). happensAt(walking(fomdgt3_6),487).
orientation(fomdgt3_1,150,488). coords(fomdgt3_1,46,102,488).  visible(fomdgt3_1,488). happensAt(inactive(fomdgt3_1),488). orientation(fomdgt3_6,177,488). coords(fomdgt3_6,42,111,488).  visible(fomdgt3_6,488). happensAt(walking(fomdgt3_6),488).
orientation(fomdgt3_1,150,489). coords(fomdgt3_1,46,102,489).  visible(fomdgt3_1,489). happensAt(inactive(fomdgt3_1),489). orientation(fomdgt3_6,177,489). coords(fomdgt3_6,42,114,489).  visible(fomdgt3_6,489). happensAt(walking(fomdgt3_6),489).
orientation(fomdgt3_1,150,490). coords(fomdgt3_1,46,102,490).  visible(fomdgt3_1,490). happensAt(inactive(fomdgt3_1),490). orientation(fomdgt3_6,177,490). coords(fomdgt3_6,42,114,490).  visible(fomdgt3_6,490). happensAt(walking(fomdgt3_6),490).
orientation(fomdgt3_1,150,491). coords(fomdgt3_1,46,102,491).  visible(fomdgt3_1,491). happensAt(inactive(fomdgt3_1),491). orientation(fomdgt3_6,177,491). coords(fomdgt3_6,42,114,491).  visible(fomdgt3_6,491). happensAt(walking(fomdgt3_6),491).
orientation(fomdgt3_1,150,492). coords(fomdgt3_1,46,102,492).  visible(fomdgt3_1,492). happensAt(inactive(fomdgt3_1),492). orientation(fomdgt3_6,177,492). coords(fomdgt3_6,42,114,492).  visible(fomdgt3_6,492). happensAt(walking(fomdgt3_6),492).
orientation(fomdgt3_1,150,493). coords(fomdgt3_1,46,102,493).  visible(fomdgt3_1,493). happensAt(inactive(fomdgt3_1),493). orientation(fomdgt3_6,177,493). coords(fomdgt3_6,42,114,493).  visible(fomdgt3_6,493). happensAt(walking(fomdgt3_6),493).
orientation(fomdgt3_1,150,494). coords(fomdgt3_1,46,102,494).  visible(fomdgt3_1,494). happensAt(inactive(fomdgt3_1),494). orientation(fomdgt3_6,177,494). coords(fomdgt3_6,42,114,494).  visible(fomdgt3_6,494). happensAt(walking(fomdgt3_6),494).
orientation(fomdgt3_1,150,495). coords(fomdgt3_1,46,102,495).  visible(fomdgt3_1,495). happensAt(inactive(fomdgt3_1),495). orientation(fomdgt3_6,177,495). coords(fomdgt3_6,42,114,495).  visible(fomdgt3_6,495). happensAt(walking(fomdgt3_6),495).
orientation(fomdgt3_1,150,496). coords(fomdgt3_1,46,102,496).  visible(fomdgt3_1,496). happensAt(inactive(fomdgt3_1),496). orientation(fomdgt3_6,177,496). coords(fomdgt3_6,42,114,496).  happensAt(disappear(fomdgt3_6),496). happensAt(walking(fomdgt3_6),496).
#show holdsAt/2.