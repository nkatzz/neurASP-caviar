
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

	orientation(wk2gt_1,125,468). coords(wk2gt_1,115,53,468).  visible(wk2gt_1,468). happensAt(inactive(wk2gt_1),468). orientation(wk2gt_2,141,468). coords(wk2gt_2,98,63,468).  visible(wk2gt_2,468). happensAt(inactive(wk2gt_2),468).
orientation(wk2gt_1,125,469). coords(wk2gt_1,115,53,469).  visible(wk2gt_1,469). happensAt(inactive(wk2gt_1),469). orientation(wk2gt_2,141,469). coords(wk2gt_2,98,63,469).  visible(wk2gt_2,469). happensAt(inactive(wk2gt_2),469).
orientation(wk2gt_1,125,470). coords(wk2gt_1,115,53,470).  visible(wk2gt_1,470). happensAt(inactive(wk2gt_1),470). orientation(wk2gt_2,141,470). coords(wk2gt_2,98,63,470).  visible(wk2gt_2,470). happensAt(inactive(wk2gt_2),470).
orientation(wk2gt_1,125,471). coords(wk2gt_1,115,53,471).  visible(wk2gt_1,471). happensAt(inactive(wk2gt_1),471). orientation(wk2gt_2,141,471). coords(wk2gt_2,98,63,471).  visible(wk2gt_2,471). happensAt(inactive(wk2gt_2),471).
orientation(wk2gt_1,125,472). coords(wk2gt_1,115,53,472).  visible(wk2gt_1,472). happensAt(inactive(wk2gt_1),472). orientation(wk2gt_2,141,472). coords(wk2gt_2,98,63,472).  visible(wk2gt_2,472). happensAt(inactive(wk2gt_2),472).
orientation(wk2gt_1,125,473). coords(wk2gt_1,115,53,473).  visible(wk2gt_1,473). happensAt(inactive(wk2gt_1),473). orientation(wk2gt_2,141,473). coords(wk2gt_2,98,63,473).  visible(wk2gt_2,473). happensAt(inactive(wk2gt_2),473).
orientation(wk2gt_1,125,474). coords(wk2gt_1,115,53,474).  visible(wk2gt_1,474). happensAt(inactive(wk2gt_1),474). orientation(wk2gt_2,141,474). coords(wk2gt_2,98,63,474).  visible(wk2gt_2,474). happensAt(inactive(wk2gt_2),474).
orientation(wk2gt_1,125,475). coords(wk2gt_1,115,53,475).  visible(wk2gt_1,475). happensAt(inactive(wk2gt_1),475). orientation(wk2gt_2,141,475). coords(wk2gt_2,98,63,475).  visible(wk2gt_2,475). happensAt(inactive(wk2gt_2),475).
orientation(wk2gt_1,125,476). coords(wk2gt_1,115,53,476).  visible(wk2gt_1,476). happensAt(inactive(wk2gt_1),476). orientation(wk2gt_2,141,476). coords(wk2gt_2,98,63,476).  visible(wk2gt_2,476). happensAt(inactive(wk2gt_2),476).
orientation(wk2gt_1,125,477). coords(wk2gt_1,115,53,477).  visible(wk2gt_1,477). happensAt(inactive(wk2gt_1),477). orientation(wk2gt_2,141,477). coords(wk2gt_2,98,63,477).  visible(wk2gt_2,477). happensAt(inactive(wk2gt_2),477).
orientation(wk2gt_1,125,478). coords(wk2gt_1,115,53,478).  visible(wk2gt_1,478). happensAt(inactive(wk2gt_1),478). orientation(wk2gt_2,141,478). coords(wk2gt_2,98,63,478).  visible(wk2gt_2,478). happensAt(inactive(wk2gt_2),478).
orientation(wk2gt_1,125,479). coords(wk2gt_1,115,53,479).  visible(wk2gt_1,479). happensAt(inactive(wk2gt_1),479). orientation(wk2gt_2,141,479). coords(wk2gt_2,98,63,479).  visible(wk2gt_2,479). happensAt(inactive(wk2gt_2),479).
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
#show holdsAt/2.