
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

	orientation(fomdgt3_1,150,458). coords(fomdgt3_1,46,102,458).  visible(fomdgt3_1,458). happensAt(inactive(fomdgt3_1),458). orientation(fomdgt3_7,150,458). coords(fomdgt3_7,100,137,458).  visible(fomdgt3_7,458). happensAt(walking(fomdgt3_7),458).
orientation(fomdgt3_1,150,459). coords(fomdgt3_1,46,102,459).  visible(fomdgt3_1,459). happensAt(inactive(fomdgt3_1),459). orientation(fomdgt3_7,150,459). coords(fomdgt3_7,100,137,459).  visible(fomdgt3_7,459). happensAt(walking(fomdgt3_7),459).
orientation(fomdgt3_1,150,460). coords(fomdgt3_1,46,102,460).  visible(fomdgt3_1,460). happensAt(inactive(fomdgt3_1),460). orientation(fomdgt3_7,150,460). coords(fomdgt3_7,101,139,460).  visible(fomdgt3_7,460). happensAt(walking(fomdgt3_7),460).
orientation(fomdgt3_1,150,461). coords(fomdgt3_1,46,102,461).  visible(fomdgt3_1,461). happensAt(inactive(fomdgt3_1),461). orientation(fomdgt3_7,150,461). coords(fomdgt3_7,101,139,461).  visible(fomdgt3_7,461). happensAt(walking(fomdgt3_7),461).
orientation(fomdgt3_1,150,462). coords(fomdgt3_1,46,102,462).  visible(fomdgt3_1,462). happensAt(inactive(fomdgt3_1),462). orientation(fomdgt3_7,150,462). coords(fomdgt3_7,101,139,462).  visible(fomdgt3_7,462). happensAt(walking(fomdgt3_7),462).
orientation(fomdgt3_1,150,463). coords(fomdgt3_1,46,102,463).  visible(fomdgt3_1,463). happensAt(inactive(fomdgt3_1),463). orientation(fomdgt3_7,150,463). coords(fomdgt3_7,101,139,463).  visible(fomdgt3_7,463). happensAt(walking(fomdgt3_7),463).
orientation(fomdgt3_1,150,464). coords(fomdgt3_1,46,102,464).  visible(fomdgt3_1,464). happensAt(inactive(fomdgt3_1),464). orientation(fomdgt3_7,150,464). coords(fomdgt3_7,101,140,464).  visible(fomdgt3_7,464). happensAt(walking(fomdgt3_7),464).
orientation(fomdgt3_1,150,465). coords(fomdgt3_1,46,102,465).  visible(fomdgt3_1,465). happensAt(inactive(fomdgt3_1),465). orientation(fomdgt3_7,150,465). coords(fomdgt3_7,101,140,465).  visible(fomdgt3_7,465). happensAt(walking(fomdgt3_7),465).
orientation(fomdgt3_1,150,466). coords(fomdgt3_1,46,102,466).  visible(fomdgt3_1,466). happensAt(inactive(fomdgt3_1),466). orientation(fomdgt3_7,150,466). coords(fomdgt3_7,101,141,466).  visible(fomdgt3_7,466). happensAt(walking(fomdgt3_7),466).
orientation(fomdgt3_1,150,467). coords(fomdgt3_1,46,102,467).  visible(fomdgt3_1,467). happensAt(inactive(fomdgt3_1),467). orientation(fomdgt3_7,150,467). coords(fomdgt3_7,103,143,467).  visible(fomdgt3_7,467). happensAt(walking(fomdgt3_7),467).
orientation(fomdgt3_1,150,468). coords(fomdgt3_1,46,102,468).  visible(fomdgt3_1,468). happensAt(inactive(fomdgt3_1),468). orientation(fomdgt3_7,150,468). coords(fomdgt3_7,104,144,468).  visible(fomdgt3_7,468). happensAt(walking(fomdgt3_7),468).
orientation(fomdgt3_1,150,469). coords(fomdgt3_1,46,102,469).  visible(fomdgt3_1,469). happensAt(inactive(fomdgt3_1),469). orientation(fomdgt3_7,150,469). coords(fomdgt3_7,105,147,469).  visible(fomdgt3_7,469). happensAt(walking(fomdgt3_7),469).
orientation(fomdgt3_1,150,470). coords(fomdgt3_1,46,102,470).  visible(fomdgt3_1,470). happensAt(inactive(fomdgt3_1),470). orientation(fomdgt3_7,150,470). coords(fomdgt3_7,106,147,470).  visible(fomdgt3_7,470). happensAt(walking(fomdgt3_7),470).
orientation(fomdgt3_1,150,471). coords(fomdgt3_1,46,102,471).  visible(fomdgt3_1,471). happensAt(inactive(fomdgt3_1),471). orientation(fomdgt3_7,150,471). coords(fomdgt3_7,106,148,471).  visible(fomdgt3_7,471). happensAt(walking(fomdgt3_7),471).
orientation(fomdgt3_1,150,472). coords(fomdgt3_1,46,102,472).  visible(fomdgt3_1,472). happensAt(inactive(fomdgt3_1),472). orientation(fomdgt3_7,150,472). coords(fomdgt3_7,107,149,472).  visible(fomdgt3_7,472). happensAt(walking(fomdgt3_7),472).
orientation(fomdgt3_1,150,473). coords(fomdgt3_1,46,102,473).  visible(fomdgt3_1,473). happensAt(inactive(fomdgt3_1),473). orientation(fomdgt3_7,150,473). coords(fomdgt3_7,109,150,473).  visible(fomdgt3_7,473). happensAt(walking(fomdgt3_7),473).
orientation(fomdgt3_1,150,474). coords(fomdgt3_1,46,102,474).  visible(fomdgt3_1,474). happensAt(inactive(fomdgt3_1),474). orientation(fomdgt3_7,150,474). coords(fomdgt3_7,110,150,474).  visible(fomdgt3_7,474). happensAt(walking(fomdgt3_7),474).
orientation(fomdgt3_1,150,475). coords(fomdgt3_1,46,102,475).  visible(fomdgt3_1,475). happensAt(inactive(fomdgt3_1),475). orientation(fomdgt3_7,150,475). coords(fomdgt3_7,110,150,475).  visible(fomdgt3_7,475). happensAt(walking(fomdgt3_7),475).
orientation(fomdgt3_1,150,476). coords(fomdgt3_1,46,102,476).  visible(fomdgt3_1,476). happensAt(inactive(fomdgt3_1),476). orientation(fomdgt3_7,150,476). coords(fomdgt3_7,110,152,476).  visible(fomdgt3_7,476). happensAt(walking(fomdgt3_7),476).
orientation(fomdgt3_1,150,477). coords(fomdgt3_1,46,102,477).  visible(fomdgt3_1,477). happensAt(inactive(fomdgt3_1),477). orientation(fomdgt3_7,150,477). coords(fomdgt3_7,110,152,477).  visible(fomdgt3_7,477). happensAt(walking(fomdgt3_7),477).
orientation(fomdgt3_1,150,478). coords(fomdgt3_1,46,102,478).  visible(fomdgt3_1,478). happensAt(inactive(fomdgt3_1),478). orientation(fomdgt3_7,150,478). coords(fomdgt3_7,110,152,478).  visible(fomdgt3_7,478). happensAt(walking(fomdgt3_7),478).
orientation(fomdgt3_1,150,479). coords(fomdgt3_1,46,102,479).  visible(fomdgt3_1,479). happensAt(inactive(fomdgt3_1),479). orientation(fomdgt3_7,150,479). coords(fomdgt3_7,110,152,479).  visible(fomdgt3_7,479). happensAt(walking(fomdgt3_7),479).
orientation(fomdgt3_1,150,480). coords(fomdgt3_1,46,102,480).  visible(fomdgt3_1,480). happensAt(inactive(fomdgt3_1),480). orientation(fomdgt3_7,150,480). coords(fomdgt3_7,110,152,480).  visible(fomdgt3_7,480). happensAt(walking(fomdgt3_7),480).
orientation(fomdgt3_1,150,481). coords(fomdgt3_1,46,102,481).  visible(fomdgt3_1,481). happensAt(inactive(fomdgt3_1),481). orientation(fomdgt3_7,159,481). coords(fomdgt3_7,110,152,481).  visible(fomdgt3_7,481). happensAt(walking(fomdgt3_7),481).
orientation(fomdgt3_1,150,482). coords(fomdgt3_1,46,102,482).  visible(fomdgt3_1,482). happensAt(inactive(fomdgt3_1),482). orientation(fomdgt3_7,159,482). coords(fomdgt3_7,113,153,482).  visible(fomdgt3_7,482). happensAt(walking(fomdgt3_7),482).
#show holdsAt/2.