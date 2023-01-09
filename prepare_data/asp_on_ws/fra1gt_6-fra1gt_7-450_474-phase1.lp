
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

	orientation(fra1gt_6,166,450). coords(fra1gt_6,45,175,450).  visible(fra1gt_6,450). happensAt(running(fra1gt_6),450). orientation(fra1gt_7,146,450). coords(fra1gt_7,57,87,450).  visible(fra1gt_7,450). happensAt(walking(fra1gt_7),450).
orientation(fra1gt_6,166,451). coords(fra1gt_6,44,176,451).  visible(fra1gt_6,451). happensAt(running(fra1gt_6),451). orientation(fra1gt_7,146,451). coords(fra1gt_7,57,88,451).  visible(fra1gt_7,451). happensAt(walking(fra1gt_7),451).
orientation(fra1gt_6,166,452). coords(fra1gt_6,44,179,452).  visible(fra1gt_6,452). happensAt(running(fra1gt_6),452). orientation(fra1gt_7,146,452). coords(fra1gt_7,56,88,452).  visible(fra1gt_7,452). happensAt(walking(fra1gt_7),452).
orientation(fra1gt_6,166,453). coords(fra1gt_6,44,180,453).  visible(fra1gt_6,453). happensAt(running(fra1gt_6),453). orientation(fra1gt_7,146,453). coords(fra1gt_7,55,88,453).  visible(fra1gt_7,453). happensAt(walking(fra1gt_7),453).
orientation(fra1gt_6,166,454). coords(fra1gt_6,42,182,454).  visible(fra1gt_6,454). happensAt(running(fra1gt_6),454). orientation(fra1gt_7,146,454). coords(fra1gt_7,55,88,454).  visible(fra1gt_7,454). happensAt(walking(fra1gt_7),454).
orientation(fra1gt_6,166,455). coords(fra1gt_6,42,182,455).  visible(fra1gt_6,455). happensAt(running(fra1gt_6),455). orientation(fra1gt_7,146,455). coords(fra1gt_7,55,89,455).  visible(fra1gt_7,455). happensAt(walking(fra1gt_7),455).
orientation(fra1gt_6,166,456). coords(fra1gt_6,41,183,456).  visible(fra1gt_6,456). happensAt(running(fra1gt_6),456). orientation(fra1gt_7,146,456). coords(fra1gt_7,54,90,456).  visible(fra1gt_7,456). happensAt(walking(fra1gt_7),456).
orientation(fra1gt_6,166,457). coords(fra1gt_6,40,185,457).  visible(fra1gt_6,457). happensAt(running(fra1gt_6),457). orientation(fra1gt_7,146,457). coords(fra1gt_7,54,90,457).  visible(fra1gt_7,457). happensAt(walking(fra1gt_7),457).
orientation(fra1gt_6,166,458). coords(fra1gt_6,39,186,458).  visible(fra1gt_6,458). happensAt(running(fra1gt_6),458). orientation(fra1gt_7,146,458). coords(fra1gt_7,54,90,458).  visible(fra1gt_7,458). happensAt(walking(fra1gt_7),458).
orientation(fra1gt_6,166,459). coords(fra1gt_6,39,188,459).  visible(fra1gt_6,459). happensAt(running(fra1gt_6),459). orientation(fra1gt_7,146,459). coords(fra1gt_7,54,90,459).  visible(fra1gt_7,459). happensAt(walking(fra1gt_7),459).
orientation(fra1gt_6,166,460). coords(fra1gt_6,38,189,460).  visible(fra1gt_6,460). happensAt(running(fra1gt_6),460). orientation(fra1gt_7,146,460). coords(fra1gt_7,54,90,460).  visible(fra1gt_7,460). happensAt(walking(fra1gt_7),460).
orientation(fra1gt_6,166,461). coords(fra1gt_6,38,190,461).  visible(fra1gt_6,461). happensAt(running(fra1gt_6),461). orientation(fra1gt_7,146,461). coords(fra1gt_7,53,90,461).  visible(fra1gt_7,461). happensAt(walking(fra1gt_7),461).
orientation(fra1gt_6,166,462). coords(fra1gt_6,38,191,462).  visible(fra1gt_6,462). happensAt(running(fra1gt_6),462). orientation(fra1gt_7,146,462). coords(fra1gt_7,53,90,462).  visible(fra1gt_7,462). happensAt(walking(fra1gt_7),462).
orientation(fra1gt_6,166,463). coords(fra1gt_6,37,192,463).  visible(fra1gt_6,463). happensAt(running(fra1gt_6),463). orientation(fra1gt_7,146,463). coords(fra1gt_7,53,90,463).  visible(fra1gt_7,463). happensAt(walking(fra1gt_7),463).
orientation(fra1gt_6,166,464). coords(fra1gt_6,36,193,464).  visible(fra1gt_6,464). happensAt(running(fra1gt_6),464). orientation(fra1gt_7,146,464). coords(fra1gt_7,51,90,464).  visible(fra1gt_7,464). happensAt(walking(fra1gt_7),464).
orientation(fra1gt_6,166,465). coords(fra1gt_6,36,194,465).  visible(fra1gt_6,465). happensAt(running(fra1gt_6),465). orientation(fra1gt_7,146,465). coords(fra1gt_7,51,90,465).  visible(fra1gt_7,465). happensAt(walking(fra1gt_7),465).
orientation(fra1gt_6,166,466). coords(fra1gt_6,36,195,466).  visible(fra1gt_6,466). happensAt(running(fra1gt_6),466). orientation(fra1gt_7,146,466). coords(fra1gt_7,51,90,466).  visible(fra1gt_7,466). happensAt(walking(fra1gt_7),466).
orientation(fra1gt_6,166,467). coords(fra1gt_6,35,196,467).  visible(fra1gt_6,467). happensAt(running(fra1gt_6),467). orientation(fra1gt_7,146,467). coords(fra1gt_7,50,90,467).  visible(fra1gt_7,467). happensAt(walking(fra1gt_7),467).
orientation(fra1gt_6,166,468). coords(fra1gt_6,35,196,468).  visible(fra1gt_6,468). happensAt(running(fra1gt_6),468). orientation(fra1gt_7,146,468). coords(fra1gt_7,50,90,468).  visible(fra1gt_7,468). happensAt(walking(fra1gt_7),468).
orientation(fra1gt_6,166,469). coords(fra1gt_6,35,197,469).  visible(fra1gt_6,469). happensAt(running(fra1gt_6),469). orientation(fra1gt_7,146,469). coords(fra1gt_7,50,90,469).  visible(fra1gt_7,469). happensAt(walking(fra1gt_7),469).
orientation(fra1gt_6,166,470). coords(fra1gt_6,35,199,470).  visible(fra1gt_6,470). happensAt(running(fra1gt_6),470). orientation(fra1gt_7,146,470). coords(fra1gt_7,50,90,470).  visible(fra1gt_7,470). happensAt(walking(fra1gt_7),470).
orientation(fra1gt_6,166,471). coords(fra1gt_6,34,199,471).  visible(fra1gt_6,471). happensAt(running(fra1gt_6),471). orientation(fra1gt_7,146,471). coords(fra1gt_7,50,90,471).  visible(fra1gt_7,471). happensAt(walking(fra1gt_7),471).
orientation(fra1gt_6,166,472). coords(fra1gt_6,34,200,472).  visible(fra1gt_6,472). happensAt(running(fra1gt_6),472). orientation(fra1gt_7,146,472). coords(fra1gt_7,50,90,472).  visible(fra1gt_7,472). happensAt(walking(fra1gt_7),472).
orientation(fra1gt_6,166,473). coords(fra1gt_6,34,202,473).  visible(fra1gt_6,473). happensAt(running(fra1gt_6),473). orientation(fra1gt_7,146,473). coords(fra1gt_7,50,90,473).  visible(fra1gt_7,473). happensAt(walking(fra1gt_7),473).
orientation(fra1gt_6,166,474). coords(fra1gt_6,32,203,474).  visible(fra1gt_6,474). happensAt(running(fra1gt_6),474). orientation(fra1gt_7,146,474). coords(fra1gt_7,50,90,474).  visible(fra1gt_7,474). happensAt(walking(fra1gt_7),474).
#show holdsAt/2.