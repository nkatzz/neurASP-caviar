
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

	orientation(mwt1gt_1,154,468). coords(mwt1gt_1,67,95,468).  visible(mwt1gt_1,468). happensAt(walking(mwt1gt_1),468). orientation(mwt1gt_2,140,468). coords(mwt1gt_2,73,79,468).  visible(mwt1gt_2,468). happensAt(walking(mwt1gt_2),468).
orientation(mwt1gt_1,154,469). coords(mwt1gt_1,67,95,469).  visible(mwt1gt_1,469). happensAt(walking(mwt1gt_1),469). orientation(mwt1gt_2,140,469). coords(mwt1gt_2,73,79,469).  visible(mwt1gt_2,469). happensAt(walking(mwt1gt_2),469).
orientation(mwt1gt_1,154,470). coords(mwt1gt_1,67,94,470).  visible(mwt1gt_1,470). happensAt(walking(mwt1gt_1),470). orientation(mwt1gt_2,140,470). coords(mwt1gt_2,73,77,470).  visible(mwt1gt_2,470). happensAt(walking(mwt1gt_2),470).
orientation(mwt1gt_1,154,471). coords(mwt1gt_1,67,94,471).  visible(mwt1gt_1,471). happensAt(walking(mwt1gt_1),471). orientation(mwt1gt_2,140,471). coords(mwt1gt_2,73,77,471).  visible(mwt1gt_2,471). happensAt(walking(mwt1gt_2),471).
orientation(mwt1gt_1,154,472). coords(mwt1gt_1,67,94,472).  visible(mwt1gt_1,472). happensAt(walking(mwt1gt_1),472). orientation(mwt1gt_2,140,472). coords(mwt1gt_2,73,77,472).  visible(mwt1gt_2,472). happensAt(walking(mwt1gt_2),472).
orientation(mwt1gt_1,154,473). coords(mwt1gt_1,66,91,473).  visible(mwt1gt_1,473). happensAt(walking(mwt1gt_1),473). orientation(mwt1gt_2,140,473). coords(mwt1gt_2,72,76,473).  visible(mwt1gt_2,473). happensAt(walking(mwt1gt_2),473).
orientation(mwt1gt_1,154,474). coords(mwt1gt_1,66,91,474).  visible(mwt1gt_1,474). happensAt(walking(mwt1gt_1),474). orientation(mwt1gt_2,140,474). coords(mwt1gt_2,72,76,474).  visible(mwt1gt_2,474). happensAt(walking(mwt1gt_2),474).
orientation(mwt1gt_1,154,475). coords(mwt1gt_1,66,88,475).  visible(mwt1gt_1,475). happensAt(walking(mwt1gt_1),475). orientation(mwt1gt_2,140,475). coords(mwt1gt_2,73,73,475).  visible(mwt1gt_2,475). happensAt(walking(mwt1gt_2),475).
orientation(mwt1gt_1,154,476). coords(mwt1gt_1,66,88,476).  visible(mwt1gt_1,476). happensAt(walking(mwt1gt_1),476). orientation(mwt1gt_2,140,476). coords(mwt1gt_2,73,72,476).  visible(mwt1gt_2,476). happensAt(walking(mwt1gt_2),476).
orientation(mwt1gt_1,154,477). coords(mwt1gt_1,66,88,477).  visible(mwt1gt_1,477). happensAt(walking(mwt1gt_1),477). orientation(mwt1gt_2,140,477). coords(mwt1gt_2,73,72,477).  visible(mwt1gt_2,477). happensAt(walking(mwt1gt_2),477).
orientation(mwt1gt_1,154,478). coords(mwt1gt_1,66,88,478).  visible(mwt1gt_1,478). happensAt(walking(mwt1gt_1),478). orientation(mwt1gt_2,140,478). coords(mwt1gt_2,73,72,478).  visible(mwt1gt_2,478). happensAt(walking(mwt1gt_2),478).
orientation(mwt1gt_1,154,479). coords(mwt1gt_1,66,88,479).  visible(mwt1gt_1,479). happensAt(walking(mwt1gt_1),479). orientation(mwt1gt_2,140,479). coords(mwt1gt_2,72,70,479).  visible(mwt1gt_2,479). happensAt(walking(mwt1gt_2),479).
orientation(mwt1gt_1,154,480). coords(mwt1gt_1,66,88,480).  visible(mwt1gt_1,480). happensAt(walking(mwt1gt_1),480). orientation(mwt1gt_2,140,480). coords(mwt1gt_2,72,70,480).  visible(mwt1gt_2,480). happensAt(walking(mwt1gt_2),480).
orientation(mwt1gt_1,154,481). coords(mwt1gt_1,66,88,481).  visible(mwt1gt_1,481). happensAt(walking(mwt1gt_1),481). orientation(mwt1gt_2,140,481). coords(mwt1gt_2,72,70,481).  visible(mwt1gt_2,481). happensAt(walking(mwt1gt_2),481).
orientation(mwt1gt_1,154,482). coords(mwt1gt_1,66,86,482).  visible(mwt1gt_1,482). happensAt(walking(mwt1gt_1),482). orientation(mwt1gt_2,140,482). coords(mwt1gt_2,72,69,482).  visible(mwt1gt_2,482). happensAt(walking(mwt1gt_2),482).
orientation(mwt1gt_1,154,483). coords(mwt1gt_1,66,86,483).  visible(mwt1gt_1,483). happensAt(walking(mwt1gt_1),483). orientation(mwt1gt_2,140,483). coords(mwt1gt_2,72,69,483).  visible(mwt1gt_2,483). happensAt(walking(mwt1gt_2),483).
orientation(mwt1gt_1,154,484). coords(mwt1gt_1,67,85,484).  visible(mwt1gt_1,484). happensAt(walking(mwt1gt_1),484). orientation(mwt1gt_2,140,484). coords(mwt1gt_2,71,67,484).  visible(mwt1gt_2,484). happensAt(walking(mwt1gt_2),484).
orientation(mwt1gt_1,154,485). coords(mwt1gt_1,67,85,485).  visible(mwt1gt_1,485). happensAt(walking(mwt1gt_1),485). orientation(mwt1gt_2,140,485). coords(mwt1gt_2,71,67,485).  visible(mwt1gt_2,485). happensAt(walking(mwt1gt_2),485).
orientation(mwt1gt_1,154,486). coords(mwt1gt_1,67,83,486).  visible(mwt1gt_1,486). happensAt(walking(mwt1gt_1),486). orientation(mwt1gt_2,140,486). coords(mwt1gt_2,74,68,486).  visible(mwt1gt_2,486). happensAt(walking(mwt1gt_2),486).
orientation(mwt1gt_1,154,487). coords(mwt1gt_1,67,79,487).  visible(mwt1gt_1,487). happensAt(walking(mwt1gt_1),487). orientation(mwt1gt_2,140,487). coords(mwt1gt_2,79,68,487).  visible(mwt1gt_2,487). happensAt(walking(mwt1gt_2),487).
orientation(mwt1gt_1,154,488). coords(mwt1gt_1,67,77,488).  visible(mwt1gt_1,488). happensAt(walking(mwt1gt_1),488). orientation(mwt1gt_2,140,488). coords(mwt1gt_2,79,67,488).  visible(mwt1gt_2,488). happensAt(walking(mwt1gt_2),488).
orientation(mwt1gt_1,154,489). coords(mwt1gt_1,67,77,489).  visible(mwt1gt_1,489). happensAt(walking(mwt1gt_1),489). orientation(mwt1gt_2,140,489). coords(mwt1gt_2,79,67,489).  visible(mwt1gt_2,489). happensAt(walking(mwt1gt_2),489).
orientation(mwt1gt_1,154,490). coords(mwt1gt_1,67,77,490).  visible(mwt1gt_1,490). happensAt(walking(mwt1gt_1),490). orientation(mwt1gt_2,140,490). coords(mwt1gt_2,80,66,490).  visible(mwt1gt_2,490). happensAt(walking(mwt1gt_2),490).
orientation(mwt1gt_1,154,491). coords(mwt1gt_1,69,76,491).  visible(mwt1gt_1,491). happensAt(walking(mwt1gt_1),491). orientation(mwt1gt_2,140,491). coords(mwt1gt_2,81,65,491).  visible(mwt1gt_2,491). happensAt(walking(mwt1gt_2),491).
orientation(mwt1gt_1,154,492). coords(mwt1gt_1,69,76,492).  visible(mwt1gt_1,492). happensAt(walking(mwt1gt_1),492). orientation(mwt1gt_2,140,492). coords(mwt1gt_2,81,65,492).  happensAt(disappear(mwt1gt_2),492). happensAt(walking(mwt1gt_2),492).
#show holdsAt/2.