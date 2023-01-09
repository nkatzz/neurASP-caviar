
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

	orientation(fomdgt3_1,150,450). coords(fomdgt3_1,46,102,450).  visible(fomdgt3_1,450). happensAt(inactive(fomdgt3_1),450). orientation(fomdgt3_6,160,450). coords(fomdgt3_6,49,103,450).  visible(fomdgt3_6,450). happensAt(walking(fomdgt3_6),450).
orientation(fomdgt3_1,150,451). coords(fomdgt3_1,46,102,451).  visible(fomdgt3_1,451). happensAt(inactive(fomdgt3_1),451). orientation(fomdgt3_6,160,451). coords(fomdgt3_6,49,103,451).  visible(fomdgt3_6,451). happensAt(walking(fomdgt3_6),451).
orientation(fomdgt3_1,150,452). coords(fomdgt3_1,46,102,452).  visible(fomdgt3_1,452). happensAt(inactive(fomdgt3_1),452). orientation(fomdgt3_6,160,452). coords(fomdgt3_6,49,103,452).  visible(fomdgt3_6,452). happensAt(walking(fomdgt3_6),452).
orientation(fomdgt3_1,150,453). coords(fomdgt3_1,46,102,453).  visible(fomdgt3_1,453). happensAt(inactive(fomdgt3_1),453). orientation(fomdgt3_6,160,453). coords(fomdgt3_6,49,103,453).  visible(fomdgt3_6,453). happensAt(walking(fomdgt3_6),453).
orientation(fomdgt3_1,150,454). coords(fomdgt3_1,46,102,454).  visible(fomdgt3_1,454). happensAt(inactive(fomdgt3_1),454). orientation(fomdgt3_6,160,454). coords(fomdgt3_6,49,103,454).  visible(fomdgt3_6,454). happensAt(walking(fomdgt3_6),454).
orientation(fomdgt3_1,150,455). coords(fomdgt3_1,46,102,455).  visible(fomdgt3_1,455). happensAt(inactive(fomdgt3_1),455). orientation(fomdgt3_6,160,455). coords(fomdgt3_6,49,103,455).  visible(fomdgt3_6,455). happensAt(walking(fomdgt3_6),455).
orientation(fomdgt3_1,150,456). coords(fomdgt3_1,46,102,456).  visible(fomdgt3_1,456). happensAt(inactive(fomdgt3_1),456). orientation(fomdgt3_6,160,456). coords(fomdgt3_6,49,103,456).  visible(fomdgt3_6,456). happensAt(walking(fomdgt3_6),456).
orientation(fomdgt3_1,150,457). coords(fomdgt3_1,46,102,457).  visible(fomdgt3_1,457). happensAt(inactive(fomdgt3_1),457). orientation(fomdgt3_6,160,457). coords(fomdgt3_6,49,103,457).  visible(fomdgt3_6,457). happensAt(walking(fomdgt3_6),457).
orientation(fomdgt3_1,150,458). coords(fomdgt3_1,46,102,458).  visible(fomdgt3_1,458). happensAt(inactive(fomdgt3_1),458). orientation(fomdgt3_6,160,458). coords(fomdgt3_6,49,103,458).  visible(fomdgt3_6,458). happensAt(walking(fomdgt3_6),458).
orientation(fomdgt3_1,150,459). coords(fomdgt3_1,46,102,459).  visible(fomdgt3_1,459). happensAt(inactive(fomdgt3_1),459). orientation(fomdgt3_6,160,459). coords(fomdgt3_6,49,103,459).  visible(fomdgt3_6,459). happensAt(walking(fomdgt3_6),459).
orientation(fomdgt3_1,150,460). coords(fomdgt3_1,46,102,460).  visible(fomdgt3_1,460). happensAt(inactive(fomdgt3_1),460). orientation(fomdgt3_6,160,460). coords(fomdgt3_6,49,103,460).  visible(fomdgt3_6,460). happensAt(walking(fomdgt3_6),460).
orientation(fomdgt3_1,150,461). coords(fomdgt3_1,46,102,461).  visible(fomdgt3_1,461). happensAt(inactive(fomdgt3_1),461). orientation(fomdgt3_6,160,461). coords(fomdgt3_6,49,103,461).  visible(fomdgt3_6,461). happensAt(walking(fomdgt3_6),461).
orientation(fomdgt3_1,150,462). coords(fomdgt3_1,46,102,462).  visible(fomdgt3_1,462). happensAt(inactive(fomdgt3_1),462). orientation(fomdgt3_6,160,462). coords(fomdgt3_6,49,103,462).  visible(fomdgt3_6,462). happensAt(walking(fomdgt3_6),462).
orientation(fomdgt3_1,150,463). coords(fomdgt3_1,46,102,463).  visible(fomdgt3_1,463). happensAt(inactive(fomdgt3_1),463). orientation(fomdgt3_6,160,463). coords(fomdgt3_6,49,103,463).  visible(fomdgt3_6,463). happensAt(walking(fomdgt3_6),463).
orientation(fomdgt3_1,150,464). coords(fomdgt3_1,46,102,464).  visible(fomdgt3_1,464). happensAt(inactive(fomdgt3_1),464). orientation(fomdgt3_6,160,464). coords(fomdgt3_6,49,103,464).  visible(fomdgt3_6,464). happensAt(walking(fomdgt3_6),464).
orientation(fomdgt3_1,150,465). coords(fomdgt3_1,46,102,465).  visible(fomdgt3_1,465). happensAt(inactive(fomdgt3_1),465). orientation(fomdgt3_6,160,465). coords(fomdgt3_6,49,103,465).  visible(fomdgt3_6,465). happensAt(walking(fomdgt3_6),465).
orientation(fomdgt3_1,150,466). coords(fomdgt3_1,46,102,466).  visible(fomdgt3_1,466). happensAt(inactive(fomdgt3_1),466). orientation(fomdgt3_6,160,466). coords(fomdgt3_6,49,103,466).  visible(fomdgt3_6,466). happensAt(walking(fomdgt3_6),466).
orientation(fomdgt3_1,150,467). coords(fomdgt3_1,46,102,467).  visible(fomdgt3_1,467). happensAt(inactive(fomdgt3_1),467). orientation(fomdgt3_6,160,467). coords(fomdgt3_6,49,103,467).  visible(fomdgt3_6,467). happensAt(walking(fomdgt3_6),467).
orientation(fomdgt3_1,150,468). coords(fomdgt3_1,46,102,468).  visible(fomdgt3_1,468). happensAt(inactive(fomdgt3_1),468). orientation(fomdgt3_6,160,468). coords(fomdgt3_6,49,103,468).  visible(fomdgt3_6,468). happensAt(walking(fomdgt3_6),468).
orientation(fomdgt3_1,150,469). coords(fomdgt3_1,46,102,469).  visible(fomdgt3_1,469). happensAt(inactive(fomdgt3_1),469). orientation(fomdgt3_6,160,469). coords(fomdgt3_6,49,103,469).  visible(fomdgt3_6,469). happensAt(walking(fomdgt3_6),469).
orientation(fomdgt3_1,150,470). coords(fomdgt3_1,46,102,470).  visible(fomdgt3_1,470). happensAt(inactive(fomdgt3_1),470). orientation(fomdgt3_6,160,470). coords(fomdgt3_6,49,103,470).  visible(fomdgt3_6,470). happensAt(walking(fomdgt3_6),470).
orientation(fomdgt3_1,150,471). coords(fomdgt3_1,46,102,471).  visible(fomdgt3_1,471). happensAt(inactive(fomdgt3_1),471). orientation(fomdgt3_6,160,471). coords(fomdgt3_6,49,103,471).  visible(fomdgt3_6,471). happensAt(walking(fomdgt3_6),471).
orientation(fomdgt3_1,150,472). coords(fomdgt3_1,46,102,472).  visible(fomdgt3_1,472). happensAt(inactive(fomdgt3_1),472). orientation(fomdgt3_6,160,472). coords(fomdgt3_6,49,103,472).  visible(fomdgt3_6,472). happensAt(walking(fomdgt3_6),472).
orientation(fomdgt3_1,150,473). coords(fomdgt3_1,46,102,473).  visible(fomdgt3_1,473). happensAt(inactive(fomdgt3_1),473). orientation(fomdgt3_6,160,473). coords(fomdgt3_6,49,103,473).  visible(fomdgt3_6,473). happensAt(walking(fomdgt3_6),473).
orientation(fomdgt3_1,150,474). coords(fomdgt3_1,46,102,474).  visible(fomdgt3_1,474). happensAt(inactive(fomdgt3_1),474). orientation(fomdgt3_6,160,474). coords(fomdgt3_6,49,103,474).  visible(fomdgt3_6,474). happensAt(walking(fomdgt3_6),474).
#show holdsAt/2.