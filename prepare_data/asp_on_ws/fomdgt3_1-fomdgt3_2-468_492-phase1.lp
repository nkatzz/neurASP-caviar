
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

	orientation(fomdgt3_1,150,468). coords(fomdgt3_1,46,102,468).  visible(fomdgt3_1,468). happensAt(inactive(fomdgt3_1),468). orientation(fomdgt3_2,42,468). coords(fomdgt3_2,262,96,468).  visible(fomdgt3_2,468). happensAt(inactive(fomdgt3_2),468).
orientation(fomdgt3_1,150,469). coords(fomdgt3_1,46,102,469).  visible(fomdgt3_1,469). happensAt(inactive(fomdgt3_1),469). orientation(fomdgt3_2,42,469). coords(fomdgt3_2,262,96,469).  visible(fomdgt3_2,469). happensAt(inactive(fomdgt3_2),469).
orientation(fomdgt3_1,150,470). coords(fomdgt3_1,46,102,470).  visible(fomdgt3_1,470). happensAt(inactive(fomdgt3_1),470). orientation(fomdgt3_2,42,470). coords(fomdgt3_2,262,96,470).  visible(fomdgt3_2,470). happensAt(inactive(fomdgt3_2),470).
orientation(fomdgt3_1,150,471). coords(fomdgt3_1,46,102,471).  visible(fomdgt3_1,471). happensAt(inactive(fomdgt3_1),471). orientation(fomdgt3_2,42,471). coords(fomdgt3_2,262,96,471).  visible(fomdgt3_2,471). happensAt(inactive(fomdgt3_2),471).
orientation(fomdgt3_1,150,472). coords(fomdgt3_1,46,102,472).  visible(fomdgt3_1,472). happensAt(inactive(fomdgt3_1),472). orientation(fomdgt3_2,42,472). coords(fomdgt3_2,262,96,472).  visible(fomdgt3_2,472). happensAt(inactive(fomdgt3_2),472).
orientation(fomdgt3_1,150,473). coords(fomdgt3_1,46,102,473).  visible(fomdgt3_1,473). happensAt(inactive(fomdgt3_1),473). orientation(fomdgt3_2,42,473). coords(fomdgt3_2,250,100,473).  visible(fomdgt3_2,473). happensAt(inactive(fomdgt3_2),473).
orientation(fomdgt3_1,150,474). coords(fomdgt3_1,46,102,474).  visible(fomdgt3_1,474). happensAt(inactive(fomdgt3_1),474). orientation(fomdgt3_2,42,474). coords(fomdgt3_2,249,101,474).  visible(fomdgt3_2,474). happensAt(inactive(fomdgt3_2),474).
orientation(fomdgt3_1,150,475). coords(fomdgt3_1,46,102,475).  visible(fomdgt3_1,475). happensAt(inactive(fomdgt3_1),475). orientation(fomdgt3_2,42,475). coords(fomdgt3_2,250,100,475).  visible(fomdgt3_2,475). happensAt(active(fomdgt3_2),475).
orientation(fomdgt3_1,150,476). coords(fomdgt3_1,46,102,476).  visible(fomdgt3_1,476). happensAt(inactive(fomdgt3_1),476). orientation(fomdgt3_2,42,476). coords(fomdgt3_2,250,100,476).  visible(fomdgt3_2,476). happensAt(active(fomdgt3_2),476).
orientation(fomdgt3_1,150,477). coords(fomdgt3_1,46,102,477).  visible(fomdgt3_1,477). happensAt(inactive(fomdgt3_1),477). orientation(fomdgt3_2,42,477). coords(fomdgt3_2,250,100,477).  visible(fomdgt3_2,477). happensAt(active(fomdgt3_2),477).
orientation(fomdgt3_1,150,478). coords(fomdgt3_1,46,102,478).  visible(fomdgt3_1,478). happensAt(inactive(fomdgt3_1),478). orientation(fomdgt3_2,42,478). coords(fomdgt3_2,250,100,478).  visible(fomdgt3_2,478). happensAt(active(fomdgt3_2),478).
orientation(fomdgt3_1,150,479). coords(fomdgt3_1,46,102,479).  visible(fomdgt3_1,479). happensAt(inactive(fomdgt3_1),479). orientation(fomdgt3_2,42,479). coords(fomdgt3_2,250,100,479).  visible(fomdgt3_2,479). happensAt(active(fomdgt3_2),479).
orientation(fomdgt3_1,150,480). coords(fomdgt3_1,46,102,480).  visible(fomdgt3_1,480). happensAt(inactive(fomdgt3_1),480). orientation(fomdgt3_2,42,480). coords(fomdgt3_2,249,98,480).  visible(fomdgt3_2,480). happensAt(active(fomdgt3_2),480).
orientation(fomdgt3_1,150,481). coords(fomdgt3_1,46,102,481).  visible(fomdgt3_1,481). happensAt(inactive(fomdgt3_1),481). orientation(fomdgt3_2,42,481). coords(fomdgt3_2,250,98,481).  visible(fomdgt3_2,481). happensAt(active(fomdgt3_2),481).
orientation(fomdgt3_1,150,482). coords(fomdgt3_1,46,102,482).  visible(fomdgt3_1,482). happensAt(inactive(fomdgt3_1),482). orientation(fomdgt3_2,42,482). coords(fomdgt3_2,250,99,482).  visible(fomdgt3_2,482). happensAt(active(fomdgt3_2),482).
orientation(fomdgt3_1,150,483). coords(fomdgt3_1,46,102,483).  visible(fomdgt3_1,483). happensAt(inactive(fomdgt3_1),483). orientation(fomdgt3_2,42,483). coords(fomdgt3_2,250,99,483).  visible(fomdgt3_2,483). happensAt(active(fomdgt3_2),483).
orientation(fomdgt3_1,150,484). coords(fomdgt3_1,46,102,484).  visible(fomdgt3_1,484). happensAt(inactive(fomdgt3_1),484). orientation(fomdgt3_2,55,484). coords(fomdgt3_2,250,99,484).  visible(fomdgt3_2,484). happensAt(active(fomdgt3_2),484).
orientation(fomdgt3_1,150,485). coords(fomdgt3_1,46,102,485).  visible(fomdgt3_1,485). happensAt(inactive(fomdgt3_1),485). orientation(fomdgt3_2,57,485). coords(fomdgt3_2,250,99,485).  visible(fomdgt3_2,485). happensAt(active(fomdgt3_2),485).
orientation(fomdgt3_1,150,486). coords(fomdgt3_1,46,102,486).  visible(fomdgt3_1,486). happensAt(inactive(fomdgt3_1),486). orientation(fomdgt3_2,57,486). coords(fomdgt3_2,251,100,486).  visible(fomdgt3_2,486). happensAt(active(fomdgt3_2),486).
orientation(fomdgt3_1,150,487). coords(fomdgt3_1,46,102,487).  visible(fomdgt3_1,487). happensAt(inactive(fomdgt3_1),487). orientation(fomdgt3_2,57,487). coords(fomdgt3_2,251,100,487).  visible(fomdgt3_2,487). happensAt(active(fomdgt3_2),487).
orientation(fomdgt3_1,150,488). coords(fomdgt3_1,46,102,488).  visible(fomdgt3_1,488). happensAt(inactive(fomdgt3_1),488). orientation(fomdgt3_2,57,488). coords(fomdgt3_2,251,100,488).  visible(fomdgt3_2,488). happensAt(active(fomdgt3_2),488).
orientation(fomdgt3_1,150,489). coords(fomdgt3_1,46,102,489).  visible(fomdgt3_1,489). happensAt(inactive(fomdgt3_1),489). orientation(fomdgt3_2,57,489). coords(fomdgt3_2,250,101,489).  visible(fomdgt3_2,489). happensAt(active(fomdgt3_2),489).
orientation(fomdgt3_1,150,490). coords(fomdgt3_1,46,102,490).  visible(fomdgt3_1,490). happensAt(inactive(fomdgt3_1),490). orientation(fomdgt3_2,57,490). coords(fomdgt3_2,250,102,490).  visible(fomdgt3_2,490). happensAt(active(fomdgt3_2),490).
orientation(fomdgt3_1,150,491). coords(fomdgt3_1,46,102,491).  visible(fomdgt3_1,491). happensAt(inactive(fomdgt3_1),491). orientation(fomdgt3_2,57,491). coords(fomdgt3_2,249,101,491).  visible(fomdgt3_2,491). happensAt(active(fomdgt3_2),491).
orientation(fomdgt3_1,150,492). coords(fomdgt3_1,46,102,492).  visible(fomdgt3_1,492). happensAt(inactive(fomdgt3_1),492). orientation(fomdgt3_2,57,492). coords(fomdgt3_2,249,101,492).  visible(fomdgt3_2,492). happensAt(walking(fomdgt3_2),492).
#show holdsAt/2.