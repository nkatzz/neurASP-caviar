
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

	orientation(ms3ggt_0,151,470). coords(ms3ggt_0,84,169,470).  visible(ms3ggt_0,470). happensAt(walking(ms3ggt_0),470). orientation(ms3ggt_1,174,470). coords(ms3ggt_1,232,208,470).  visible(ms3ggt_1,470). happensAt(walking(ms3ggt_1),470).
orientation(ms3ggt_0,151,471). coords(ms3ggt_0,84,169,471).  visible(ms3ggt_0,471). happensAt(walking(ms3ggt_0),471). orientation(ms3ggt_1,174,471). coords(ms3ggt_1,235,211,471).  visible(ms3ggt_1,471). happensAt(walking(ms3ggt_1),471).
orientation(ms3ggt_0,151,472). coords(ms3ggt_0,83,170,472).  visible(ms3ggt_0,472). happensAt(walking(ms3ggt_0),472). orientation(ms3ggt_1,174,472). coords(ms3ggt_1,237,213,472).  visible(ms3ggt_1,472). happensAt(walking(ms3ggt_1),472).
orientation(ms3ggt_0,151,473). coords(ms3ggt_0,81,170,473).  visible(ms3ggt_0,473). happensAt(walking(ms3ggt_0),473). orientation(ms3ggt_1,177,473). coords(ms3ggt_1,240,216,473).  visible(ms3ggt_1,473). happensAt(walking(ms3ggt_1),473).
orientation(ms3ggt_0,151,474). coords(ms3ggt_0,80,171,474).  visible(ms3ggt_0,474). happensAt(walking(ms3ggt_0),474). orientation(ms3ggt_1,177,474). coords(ms3ggt_1,243,218,474).  visible(ms3ggt_1,474). happensAt(walking(ms3ggt_1),474).
orientation(ms3ggt_0,151,475). coords(ms3ggt_0,79,172,475).  visible(ms3ggt_0,475). happensAt(walking(ms3ggt_0),475). orientation(ms3ggt_1,177,475). coords(ms3ggt_1,244,220,475).  visible(ms3ggt_1,475). happensAt(walking(ms3ggt_1),475).
orientation(ms3ggt_0,151,476). coords(ms3ggt_0,79,172,476).  visible(ms3ggt_0,476). happensAt(walking(ms3ggt_0),476). orientation(ms3ggt_1,177,476). coords(ms3ggt_1,246,221,476).  visible(ms3ggt_1,476). happensAt(walking(ms3ggt_1),476).
orientation(ms3ggt_0,151,477). coords(ms3ggt_0,79,172,477).  visible(ms3ggt_0,477). happensAt(walking(ms3ggt_0),477). orientation(ms3ggt_1,177,477). coords(ms3ggt_1,246,222,477).  visible(ms3ggt_1,477). happensAt(walking(ms3ggt_1),477).
orientation(ms3ggt_0,151,478). coords(ms3ggt_0,79,172,478).  visible(ms3ggt_0,478). happensAt(walking(ms3ggt_0),478). orientation(ms3ggt_1,177,478). coords(ms3ggt_1,247,223,478).  visible(ms3ggt_1,478). happensAt(walking(ms3ggt_1),478).
orientation(ms3ggt_0,151,479). coords(ms3ggt_0,79,172,479).  visible(ms3ggt_0,479). happensAt(walking(ms3ggt_0),479). orientation(ms3ggt_1,5,479). coords(ms3ggt_1,248,224,479).  visible(ms3ggt_1,479). happensAt(walking(ms3ggt_1),479).
orientation(ms3ggt_0,151,480). coords(ms3ggt_0,79,172,480).  visible(ms3ggt_0,480). happensAt(walking(ms3ggt_0),480). orientation(ms3ggt_1,14,480). coords(ms3ggt_1,248,224,480).  visible(ms3ggt_1,480). happensAt(walking(ms3ggt_1),480).
orientation(ms3ggt_0,156,481). coords(ms3ggt_0,78,178,481).  visible(ms3ggt_0,481). happensAt(walking(ms3ggt_0),481). orientation(ms3ggt_1,14,481). coords(ms3ggt_1,249,226,481).  visible(ms3ggt_1,481). happensAt(walking(ms3ggt_1),481).
orientation(ms3ggt_0,156,482). coords(ms3ggt_0,78,179,482).  visible(ms3ggt_0,482). happensAt(walking(ms3ggt_0),482). orientation(ms3ggt_1,14,482). coords(ms3ggt_1,249,228,482).  visible(ms3ggt_1,482). happensAt(walking(ms3ggt_1),482).
orientation(ms3ggt_0,156,483). coords(ms3ggt_0,78,181,483).  visible(ms3ggt_0,483). happensAt(walking(ms3ggt_0),483). orientation(ms3ggt_1,14,483). coords(ms3ggt_1,250,231,483).  visible(ms3ggt_1,483). happensAt(walking(ms3ggt_1),483).
orientation(ms3ggt_0,156,484). coords(ms3ggt_0,78,181,484).  visible(ms3ggt_0,484). happensAt(walking(ms3ggt_0),484). orientation(ms3ggt_1,14,484). coords(ms3ggt_1,250,235,484).  visible(ms3ggt_1,484). happensAt(walking(ms3ggt_1),484).
orientation(ms3ggt_0,156,485). coords(ms3ggt_0,78,182,485).  visible(ms3ggt_0,485). happensAt(walking(ms3ggt_0),485). orientation(ms3ggt_1,18,485). coords(ms3ggt_1,251,238,485).  visible(ms3ggt_1,485). happensAt(walking(ms3ggt_1),485).
orientation(ms3ggt_0,156,486). coords(ms3ggt_0,77,182,486).  visible(ms3ggt_0,486). happensAt(walking(ms3ggt_0),486). orientation(ms3ggt_1,18,486). coords(ms3ggt_1,252,240,486).  visible(ms3ggt_1,486). happensAt(walking(ms3ggt_1),486).
orientation(ms3ggt_0,156,487). coords(ms3ggt_0,77,182,487).  visible(ms3ggt_0,487). happensAt(walking(ms3ggt_0),487). orientation(ms3ggt_1,22,487). coords(ms3ggt_1,253,242,487).  visible(ms3ggt_1,487). happensAt(walking(ms3ggt_1),487).
orientation(ms3ggt_0,160,488). coords(ms3ggt_0,76,183,488).  visible(ms3ggt_0,488). happensAt(walking(ms3ggt_0),488). orientation(ms3ggt_1,22,488). coords(ms3ggt_1,255,244,488).  visible(ms3ggt_1,488). happensAt(walking(ms3ggt_1),488).
orientation(ms3ggt_0,160,489). coords(ms3ggt_0,76,184,489).  visible(ms3ggt_0,489). happensAt(walking(ms3ggt_0),489). orientation(ms3ggt_1,22,489). coords(ms3ggt_1,255,246,489).  visible(ms3ggt_1,489). happensAt(walking(ms3ggt_1),489).
orientation(ms3ggt_0,160,490). coords(ms3ggt_0,76,184,490).  visible(ms3ggt_0,490). happensAt(walking(ms3ggt_0),490). orientation(ms3ggt_1,22,490). coords(ms3ggt_1,256,248,490).  visible(ms3ggt_1,490). happensAt(walking(ms3ggt_1),490).
orientation(ms3ggt_0,163,491). coords(ms3ggt_0,75,185,491).  visible(ms3ggt_0,491). happensAt(walking(ms3ggt_0),491). orientation(ms3ggt_1,22,491). coords(ms3ggt_1,256,248,491).  visible(ms3ggt_1,491). happensAt(walking(ms3ggt_1),491).
orientation(ms3ggt_0,163,492). coords(ms3ggt_0,75,185,492).  visible(ms3ggt_0,492). happensAt(walking(ms3ggt_0),492). orientation(ms3ggt_1,22,492). coords(ms3ggt_1,256,248,492).  visible(ms3ggt_1,492). happensAt(walking(ms3ggt_1),492).
orientation(ms3ggt_0,163,493). coords(ms3ggt_0,75,186,493).  visible(ms3ggt_0,493). happensAt(walking(ms3ggt_0),493). orientation(ms3ggt_1,22,493). coords(ms3ggt_1,256,248,493).  visible(ms3ggt_1,493). happensAt(walking(ms3ggt_1),493).
orientation(ms3ggt_0,163,494). coords(ms3ggt_0,75,186,494).  visible(ms3ggt_0,494). happensAt(walking(ms3ggt_0),494). orientation(ms3ggt_1,29,494). coords(ms3ggt_1,256,250,494).  visible(ms3ggt_1,494). happensAt(walking(ms3ggt_1),494).
#show holdsAt/2.