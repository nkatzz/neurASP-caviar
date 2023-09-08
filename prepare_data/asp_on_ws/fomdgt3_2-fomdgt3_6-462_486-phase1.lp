
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

	orientation(fomdgt3_2,42,462). coords(fomdgt3_2,262,96,462).  visible(fomdgt3_2,462). happensAt(inactive(fomdgt3_2),462). orientation(fomdgt3_6,160,462). coords(fomdgt3_6,49,103,462).  visible(fomdgt3_6,462). happensAt(walking(fomdgt3_6),462).
orientation(fomdgt3_2,42,463). coords(fomdgt3_2,262,96,463).  visible(fomdgt3_2,463). happensAt(inactive(fomdgt3_2),463). orientation(fomdgt3_6,160,463). coords(fomdgt3_6,49,103,463).  visible(fomdgt3_6,463). happensAt(walking(fomdgt3_6),463).
orientation(fomdgt3_2,42,464). coords(fomdgt3_2,262,96,464).  visible(fomdgt3_2,464). happensAt(inactive(fomdgt3_2),464). orientation(fomdgt3_6,160,464). coords(fomdgt3_6,49,103,464).  visible(fomdgt3_6,464). happensAt(walking(fomdgt3_6),464).
orientation(fomdgt3_2,42,465). coords(fomdgt3_2,262,96,465).  visible(fomdgt3_2,465). happensAt(inactive(fomdgt3_2),465). orientation(fomdgt3_6,160,465). coords(fomdgt3_6,49,103,465).  visible(fomdgt3_6,465). happensAt(walking(fomdgt3_6),465).
orientation(fomdgt3_2,42,466). coords(fomdgt3_2,262,96,466).  visible(fomdgt3_2,466). happensAt(inactive(fomdgt3_2),466). orientation(fomdgt3_6,160,466). coords(fomdgt3_6,49,103,466).  visible(fomdgt3_6,466). happensAt(walking(fomdgt3_6),466).
orientation(fomdgt3_2,42,467). coords(fomdgt3_2,262,96,467).  visible(fomdgt3_2,467). happensAt(inactive(fomdgt3_2),467). orientation(fomdgt3_6,160,467). coords(fomdgt3_6,49,103,467).  visible(fomdgt3_6,467). happensAt(walking(fomdgt3_6),467).
orientation(fomdgt3_2,42,468). coords(fomdgt3_2,262,96,468).  visible(fomdgt3_2,468). happensAt(inactive(fomdgt3_2),468). orientation(fomdgt3_6,160,468). coords(fomdgt3_6,49,103,468).  visible(fomdgt3_6,468). happensAt(walking(fomdgt3_6),468).
orientation(fomdgt3_2,42,469). coords(fomdgt3_2,262,96,469).  visible(fomdgt3_2,469). happensAt(inactive(fomdgt3_2),469). orientation(fomdgt3_6,160,469). coords(fomdgt3_6,49,103,469).  visible(fomdgt3_6,469). happensAt(walking(fomdgt3_6),469).
orientation(fomdgt3_2,42,470). coords(fomdgt3_2,262,96,470).  visible(fomdgt3_2,470). happensAt(inactive(fomdgt3_2),470). orientation(fomdgt3_6,160,470). coords(fomdgt3_6,49,103,470).  visible(fomdgt3_6,470). happensAt(walking(fomdgt3_6),470).
orientation(fomdgt3_2,42,471). coords(fomdgt3_2,262,96,471).  visible(fomdgt3_2,471). happensAt(inactive(fomdgt3_2),471). orientation(fomdgt3_6,160,471). coords(fomdgt3_6,49,103,471).  visible(fomdgt3_6,471). happensAt(walking(fomdgt3_6),471).
orientation(fomdgt3_2,42,472). coords(fomdgt3_2,262,96,472).  visible(fomdgt3_2,472). happensAt(inactive(fomdgt3_2),472). orientation(fomdgt3_6,160,472). coords(fomdgt3_6,49,103,472).  visible(fomdgt3_6,472). happensAt(walking(fomdgt3_6),472).
orientation(fomdgt3_2,42,473). coords(fomdgt3_2,250,100,473).  visible(fomdgt3_2,473). happensAt(inactive(fomdgt3_2),473). orientation(fomdgt3_6,160,473). coords(fomdgt3_6,49,103,473).  visible(fomdgt3_6,473). happensAt(walking(fomdgt3_6),473).
orientation(fomdgt3_2,42,474). coords(fomdgt3_2,249,101,474).  visible(fomdgt3_2,474). happensAt(inactive(fomdgt3_2),474). orientation(fomdgt3_6,160,474). coords(fomdgt3_6,49,103,474).  visible(fomdgt3_6,474). happensAt(walking(fomdgt3_6),474).
orientation(fomdgt3_2,42,475). coords(fomdgt3_2,250,100,475).  visible(fomdgt3_2,475). happensAt(active(fomdgt3_2),475). orientation(fomdgt3_6,160,475). coords(fomdgt3_6,49,103,475).  visible(fomdgt3_6,475). happensAt(walking(fomdgt3_6),475).
orientation(fomdgt3_2,42,476). coords(fomdgt3_2,250,100,476).  visible(fomdgt3_2,476). happensAt(active(fomdgt3_2),476). orientation(fomdgt3_6,160,476). coords(fomdgt3_6,49,103,476).  visible(fomdgt3_6,476). happensAt(walking(fomdgt3_6),476).
orientation(fomdgt3_2,42,477). coords(fomdgt3_2,250,100,477).  visible(fomdgt3_2,477). happensAt(active(fomdgt3_2),477). orientation(fomdgt3_6,160,477). coords(fomdgt3_6,49,103,477).  visible(fomdgt3_6,477). happensAt(walking(fomdgt3_6),477).
orientation(fomdgt3_2,42,478). coords(fomdgt3_2,250,100,478).  visible(fomdgt3_2,478). happensAt(active(fomdgt3_2),478). orientation(fomdgt3_6,160,478). coords(fomdgt3_6,49,103,478).  visible(fomdgt3_6,478). happensAt(walking(fomdgt3_6),478).
orientation(fomdgt3_2,42,479). coords(fomdgt3_2,250,100,479).  visible(fomdgt3_2,479). happensAt(active(fomdgt3_2),479). orientation(fomdgt3_6,160,479). coords(fomdgt3_6,49,103,479).  visible(fomdgt3_6,479). happensAt(walking(fomdgt3_6),479).
orientation(fomdgt3_2,42,480). coords(fomdgt3_2,249,98,480).  visible(fomdgt3_2,480). happensAt(active(fomdgt3_2),480). orientation(fomdgt3_6,160,480). coords(fomdgt3_6,49,103,480).  visible(fomdgt3_6,480). happensAt(walking(fomdgt3_6),480).
orientation(fomdgt3_2,42,481). coords(fomdgt3_2,250,98,481).  visible(fomdgt3_2,481). happensAt(active(fomdgt3_2),481). orientation(fomdgt3_6,160,481). coords(fomdgt3_6,49,103,481).  visible(fomdgt3_6,481). happensAt(walking(fomdgt3_6),481).
orientation(fomdgt3_2,42,482). coords(fomdgt3_2,250,99,482).  visible(fomdgt3_2,482). happensAt(active(fomdgt3_2),482). orientation(fomdgt3_6,160,482). coords(fomdgt3_6,49,103,482).  visible(fomdgt3_6,482). happensAt(walking(fomdgt3_6),482).
orientation(fomdgt3_2,42,483). coords(fomdgt3_2,250,99,483).  visible(fomdgt3_2,483). happensAt(active(fomdgt3_2),483). orientation(fomdgt3_6,160,483). coords(fomdgt3_6,49,103,483).  visible(fomdgt3_6,483). happensAt(walking(fomdgt3_6),483).
orientation(fomdgt3_2,55,484). coords(fomdgt3_2,250,99,484).  visible(fomdgt3_2,484). happensAt(active(fomdgt3_2),484). orientation(fomdgt3_6,177,484). coords(fomdgt3_6,42,108,484).  visible(fomdgt3_6,484). happensAt(walking(fomdgt3_6),484).
orientation(fomdgt3_2,57,485). coords(fomdgt3_2,250,99,485).  visible(fomdgt3_2,485). happensAt(active(fomdgt3_2),485). orientation(fomdgt3_6,177,485). coords(fomdgt3_6,42,108,485).  visible(fomdgt3_6,485). happensAt(walking(fomdgt3_6),485).
orientation(fomdgt3_2,57,486). coords(fomdgt3_2,251,100,486).  visible(fomdgt3_2,486). happensAt(active(fomdgt3_2),486). orientation(fomdgt3_6,177,486). coords(fomdgt3_6,42,111,486).  visible(fomdgt3_6,486). happensAt(walking(fomdgt3_6),486).
#show holdsAt/2.