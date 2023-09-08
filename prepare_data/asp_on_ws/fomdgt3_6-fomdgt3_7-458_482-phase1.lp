
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

	orientation(fomdgt3_6,160,458). coords(fomdgt3_6,49,103,458).  visible(fomdgt3_6,458). happensAt(walking(fomdgt3_6),458). orientation(fomdgt3_7,150,458). coords(fomdgt3_7,100,137,458).  visible(fomdgt3_7,458). happensAt(walking(fomdgt3_7),458).
orientation(fomdgt3_6,160,459). coords(fomdgt3_6,49,103,459).  visible(fomdgt3_6,459). happensAt(walking(fomdgt3_6),459). orientation(fomdgt3_7,150,459). coords(fomdgt3_7,100,137,459).  visible(fomdgt3_7,459). happensAt(walking(fomdgt3_7),459).
orientation(fomdgt3_6,160,460). coords(fomdgt3_6,49,103,460).  visible(fomdgt3_6,460). happensAt(walking(fomdgt3_6),460). orientation(fomdgt3_7,150,460). coords(fomdgt3_7,101,139,460).  visible(fomdgt3_7,460). happensAt(walking(fomdgt3_7),460).
orientation(fomdgt3_6,160,461). coords(fomdgt3_6,49,103,461).  visible(fomdgt3_6,461). happensAt(walking(fomdgt3_6),461). orientation(fomdgt3_7,150,461). coords(fomdgt3_7,101,139,461).  visible(fomdgt3_7,461). happensAt(walking(fomdgt3_7),461).
orientation(fomdgt3_6,160,462). coords(fomdgt3_6,49,103,462).  visible(fomdgt3_6,462). happensAt(walking(fomdgt3_6),462). orientation(fomdgt3_7,150,462). coords(fomdgt3_7,101,139,462).  visible(fomdgt3_7,462). happensAt(walking(fomdgt3_7),462).
orientation(fomdgt3_6,160,463). coords(fomdgt3_6,49,103,463).  visible(fomdgt3_6,463). happensAt(walking(fomdgt3_6),463). orientation(fomdgt3_7,150,463). coords(fomdgt3_7,101,139,463).  visible(fomdgt3_7,463). happensAt(walking(fomdgt3_7),463).
orientation(fomdgt3_6,160,464). coords(fomdgt3_6,49,103,464).  visible(fomdgt3_6,464). happensAt(walking(fomdgt3_6),464). orientation(fomdgt3_7,150,464). coords(fomdgt3_7,101,140,464).  visible(fomdgt3_7,464). happensAt(walking(fomdgt3_7),464).
orientation(fomdgt3_6,160,465). coords(fomdgt3_6,49,103,465).  visible(fomdgt3_6,465). happensAt(walking(fomdgt3_6),465). orientation(fomdgt3_7,150,465). coords(fomdgt3_7,101,140,465).  visible(fomdgt3_7,465). happensAt(walking(fomdgt3_7),465).
orientation(fomdgt3_6,160,466). coords(fomdgt3_6,49,103,466).  visible(fomdgt3_6,466). happensAt(walking(fomdgt3_6),466). orientation(fomdgt3_7,150,466). coords(fomdgt3_7,101,141,466).  visible(fomdgt3_7,466). happensAt(walking(fomdgt3_7),466).
orientation(fomdgt3_6,160,467). coords(fomdgt3_6,49,103,467).  visible(fomdgt3_6,467). happensAt(walking(fomdgt3_6),467). orientation(fomdgt3_7,150,467). coords(fomdgt3_7,103,143,467).  visible(fomdgt3_7,467). happensAt(walking(fomdgt3_7),467).
orientation(fomdgt3_6,160,468). coords(fomdgt3_6,49,103,468).  visible(fomdgt3_6,468). happensAt(walking(fomdgt3_6),468). orientation(fomdgt3_7,150,468). coords(fomdgt3_7,104,144,468).  visible(fomdgt3_7,468). happensAt(walking(fomdgt3_7),468).
orientation(fomdgt3_6,160,469). coords(fomdgt3_6,49,103,469).  visible(fomdgt3_6,469). happensAt(walking(fomdgt3_6),469). orientation(fomdgt3_7,150,469). coords(fomdgt3_7,105,147,469).  visible(fomdgt3_7,469). happensAt(walking(fomdgt3_7),469).
orientation(fomdgt3_6,160,470). coords(fomdgt3_6,49,103,470).  visible(fomdgt3_6,470). happensAt(walking(fomdgt3_6),470). orientation(fomdgt3_7,150,470). coords(fomdgt3_7,106,147,470).  visible(fomdgt3_7,470). happensAt(walking(fomdgt3_7),470).
orientation(fomdgt3_6,160,471). coords(fomdgt3_6,49,103,471).  visible(fomdgt3_6,471). happensAt(walking(fomdgt3_6),471). orientation(fomdgt3_7,150,471). coords(fomdgt3_7,106,148,471).  visible(fomdgt3_7,471). happensAt(walking(fomdgt3_7),471).
orientation(fomdgt3_6,160,472). coords(fomdgt3_6,49,103,472).  visible(fomdgt3_6,472). happensAt(walking(fomdgt3_6),472). orientation(fomdgt3_7,150,472). coords(fomdgt3_7,107,149,472).  visible(fomdgt3_7,472). happensAt(walking(fomdgt3_7),472).
orientation(fomdgt3_6,160,473). coords(fomdgt3_6,49,103,473).  visible(fomdgt3_6,473). happensAt(walking(fomdgt3_6),473). orientation(fomdgt3_7,150,473). coords(fomdgt3_7,109,150,473).  visible(fomdgt3_7,473). happensAt(walking(fomdgt3_7),473).
orientation(fomdgt3_6,160,474). coords(fomdgt3_6,49,103,474).  visible(fomdgt3_6,474). happensAt(walking(fomdgt3_6),474). orientation(fomdgt3_7,150,474). coords(fomdgt3_7,110,150,474).  visible(fomdgt3_7,474). happensAt(walking(fomdgt3_7),474).
orientation(fomdgt3_6,160,475). coords(fomdgt3_6,49,103,475).  visible(fomdgt3_6,475). happensAt(walking(fomdgt3_6),475). orientation(fomdgt3_7,150,475). coords(fomdgt3_7,110,150,475).  visible(fomdgt3_7,475). happensAt(walking(fomdgt3_7),475).
orientation(fomdgt3_6,160,476). coords(fomdgt3_6,49,103,476).  visible(fomdgt3_6,476). happensAt(walking(fomdgt3_6),476). orientation(fomdgt3_7,150,476). coords(fomdgt3_7,110,152,476).  visible(fomdgt3_7,476). happensAt(walking(fomdgt3_7),476).
orientation(fomdgt3_6,160,477). coords(fomdgt3_6,49,103,477).  visible(fomdgt3_6,477). happensAt(walking(fomdgt3_6),477). orientation(fomdgt3_7,150,477). coords(fomdgt3_7,110,152,477).  visible(fomdgt3_7,477). happensAt(walking(fomdgt3_7),477).
orientation(fomdgt3_6,160,478). coords(fomdgt3_6,49,103,478).  visible(fomdgt3_6,478). happensAt(walking(fomdgt3_6),478). orientation(fomdgt3_7,150,478). coords(fomdgt3_7,110,152,478).  visible(fomdgt3_7,478). happensAt(walking(fomdgt3_7),478).
orientation(fomdgt3_6,160,479). coords(fomdgt3_6,49,103,479).  visible(fomdgt3_6,479). happensAt(walking(fomdgt3_6),479). orientation(fomdgt3_7,150,479). coords(fomdgt3_7,110,152,479).  visible(fomdgt3_7,479). happensAt(walking(fomdgt3_7),479).
orientation(fomdgt3_6,160,480). coords(fomdgt3_6,49,103,480).  visible(fomdgt3_6,480). happensAt(walking(fomdgt3_6),480). orientation(fomdgt3_7,150,480). coords(fomdgt3_7,110,152,480).  visible(fomdgt3_7,480). happensAt(walking(fomdgt3_7),480).
orientation(fomdgt3_6,160,481). coords(fomdgt3_6,49,103,481).  visible(fomdgt3_6,481). happensAt(walking(fomdgt3_6),481). orientation(fomdgt3_7,159,481). coords(fomdgt3_7,110,152,481).  visible(fomdgt3_7,481). happensAt(walking(fomdgt3_7),481).
orientation(fomdgt3_6,160,482). coords(fomdgt3_6,49,103,482).  visible(fomdgt3_6,482). happensAt(walking(fomdgt3_6),482). orientation(fomdgt3_7,159,482). coords(fomdgt3_7,113,153,482).  visible(fomdgt3_7,482). happensAt(walking(fomdgt3_7),482).
#show holdsAt/2.