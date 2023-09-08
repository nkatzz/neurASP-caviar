
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

	orientation(wk1gt_4,147,447). coords(wk1gt_4,56,169,447).  visible(wk1gt_4,447). happensAt(active(wk1gt_4),447). orientation(wk1gt_5,162,447). coords(wk1gt_5,54,159,447).  visible(wk1gt_5,447). happensAt(walking(wk1gt_5),447).
orientation(wk1gt_4,147,448). coords(wk1gt_4,56,169,448).  visible(wk1gt_4,448). happensAt(active(wk1gt_4),448). orientation(wk1gt_5,162,448). coords(wk1gt_5,54,159,448).  visible(wk1gt_5,448). happensAt(walking(wk1gt_5),448).
orientation(wk1gt_4,147,449). coords(wk1gt_4,56,169,449).  visible(wk1gt_4,449). happensAt(active(wk1gt_4),449). orientation(wk1gt_5,162,449). coords(wk1gt_5,54,159,449).  visible(wk1gt_5,449). happensAt(walking(wk1gt_5),449).
orientation(wk1gt_4,147,450). coords(wk1gt_4,56,169,450).  visible(wk1gt_4,450). happensAt(active(wk1gt_4),450). orientation(wk1gt_5,162,450). coords(wk1gt_5,54,158,450).  visible(wk1gt_5,450). happensAt(walking(wk1gt_5),450).
orientation(wk1gt_4,147,451). coords(wk1gt_4,56,169,451).  visible(wk1gt_4,451). happensAt(active(wk1gt_4),451). orientation(wk1gt_5,162,451). coords(wk1gt_5,54,158,451).  visible(wk1gt_5,451). happensAt(walking(wk1gt_5),451).
orientation(wk1gt_4,147,452). coords(wk1gt_4,56,169,452).  visible(wk1gt_4,452). happensAt(active(wk1gt_4),452). orientation(wk1gt_5,162,452). coords(wk1gt_5,54,158,452).  visible(wk1gt_5,452). happensAt(walking(wk1gt_5),452).
orientation(wk1gt_4,147,453). coords(wk1gt_4,56,169,453).  visible(wk1gt_4,453). happensAt(active(wk1gt_4),453). orientation(wk1gt_5,162,453). coords(wk1gt_5,54,158,453).  visible(wk1gt_5,453). happensAt(walking(wk1gt_5),453).
orientation(wk1gt_4,147,454). coords(wk1gt_4,56,169,454).  visible(wk1gt_4,454). happensAt(active(wk1gt_4),454). orientation(wk1gt_5,162,454). coords(wk1gt_5,54,158,454).  visible(wk1gt_5,454). happensAt(walking(wk1gt_5),454).
orientation(wk1gt_4,147,455). coords(wk1gt_4,56,169,455).  visible(wk1gt_4,455). happensAt(active(wk1gt_4),455). orientation(wk1gt_5,162,455). coords(wk1gt_5,54,158,455).  visible(wk1gt_5,455). happensAt(walking(wk1gt_5),455).
orientation(wk1gt_4,147,456). coords(wk1gt_4,56,169,456).  visible(wk1gt_4,456). happensAt(active(wk1gt_4),456). orientation(wk1gt_5,162,456). coords(wk1gt_5,54,158,456).  visible(wk1gt_5,456). happensAt(walking(wk1gt_5),456).
orientation(wk1gt_4,147,457). coords(wk1gt_4,56,169,457).  visible(wk1gt_4,457). happensAt(active(wk1gt_4),457). orientation(wk1gt_5,162,457). coords(wk1gt_5,54,156,457).  visible(wk1gt_5,457). happensAt(walking(wk1gt_5),457).
orientation(wk1gt_4,147,458). coords(wk1gt_4,56,169,458).  visible(wk1gt_4,458). happensAt(active(wk1gt_4),458). orientation(wk1gt_5,162,458). coords(wk1gt_5,55,156,458).  visible(wk1gt_5,458). happensAt(walking(wk1gt_5),458).
orientation(wk1gt_4,147,459). coords(wk1gt_4,56,169,459).  visible(wk1gt_4,459). happensAt(active(wk1gt_4),459). orientation(wk1gt_5,162,459). coords(wk1gt_5,55,156,459).  visible(wk1gt_5,459). happensAt(walking(wk1gt_5),459).
orientation(wk1gt_4,147,460). coords(wk1gt_4,56,169,460).  visible(wk1gt_4,460). happensAt(active(wk1gt_4),460). orientation(wk1gt_5,162,460). coords(wk1gt_5,55,155,460).  visible(wk1gt_5,460). happensAt(walking(wk1gt_5),460).
orientation(wk1gt_4,147,461). coords(wk1gt_4,56,169,461).  visible(wk1gt_4,461). happensAt(active(wk1gt_4),461). orientation(wk1gt_5,162,461). coords(wk1gt_5,55,155,461).  visible(wk1gt_5,461). happensAt(walking(wk1gt_5),461).
orientation(wk1gt_4,147,462). coords(wk1gt_4,56,169,462).  visible(wk1gt_4,462). happensAt(active(wk1gt_4),462). orientation(wk1gt_5,162,462). coords(wk1gt_5,56,154,462).  visible(wk1gt_5,462). happensAt(walking(wk1gt_5),462).
orientation(wk1gt_4,147,463). coords(wk1gt_4,56,169,463).  visible(wk1gt_4,463). happensAt(active(wk1gt_4),463). orientation(wk1gt_5,162,463). coords(wk1gt_5,56,154,463).  visible(wk1gt_5,463). happensAt(walking(wk1gt_5),463).
orientation(wk1gt_4,147,464). coords(wk1gt_4,56,169,464).  visible(wk1gt_4,464). happensAt(active(wk1gt_4),464). orientation(wk1gt_5,162,464). coords(wk1gt_5,56,154,464).  visible(wk1gt_5,464). happensAt(walking(wk1gt_5),464).
orientation(wk1gt_4,147,465). coords(wk1gt_4,56,169,465).  visible(wk1gt_4,465). happensAt(active(wk1gt_4),465). orientation(wk1gt_5,162,465). coords(wk1gt_5,56,153,465).  visible(wk1gt_5,465). happensAt(walking(wk1gt_5),465).
orientation(wk1gt_4,147,466). coords(wk1gt_4,56,169,466).  visible(wk1gt_4,466). happensAt(active(wk1gt_4),466). orientation(wk1gt_5,162,466). coords(wk1gt_5,56,153,466).  visible(wk1gt_5,466). happensAt(walking(wk1gt_5),466).
orientation(wk1gt_4,147,467). coords(wk1gt_4,56,169,467).  visible(wk1gt_4,467). happensAt(active(wk1gt_4),467). orientation(wk1gt_5,162,467). coords(wk1gt_5,56,153,467).  visible(wk1gt_5,467). happensAt(walking(wk1gt_5),467).
orientation(wk1gt_4,147,468). coords(wk1gt_4,56,169,468).  visible(wk1gt_4,468). happensAt(active(wk1gt_4),468). orientation(wk1gt_5,162,468). coords(wk1gt_5,56,153,468).  visible(wk1gt_5,468). happensAt(walking(wk1gt_5),468).
orientation(wk1gt_4,147,469). coords(wk1gt_4,56,168,469).  visible(wk1gt_4,469). happensAt(active(wk1gt_4),469). orientation(wk1gt_5,162,469). coords(wk1gt_5,56,153,469).  visible(wk1gt_5,469). happensAt(walking(wk1gt_5),469).
orientation(wk1gt_4,147,470). coords(wk1gt_4,56,168,470).  visible(wk1gt_4,470). happensAt(active(wk1gt_4),470). orientation(wk1gt_5,162,470). coords(wk1gt_5,56,153,470).  visible(wk1gt_5,470). happensAt(walking(wk1gt_5),470).
orientation(wk1gt_4,147,471). coords(wk1gt_4,56,168,471).  visible(wk1gt_4,471). happensAt(active(wk1gt_4),471). orientation(wk1gt_5,162,471). coords(wk1gt_5,57,153,471).  visible(wk1gt_5,471). happensAt(walking(wk1gt_5),471).
#show holdsAt/2.