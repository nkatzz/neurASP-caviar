
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

	orientation(lb1gt_1,3,444). coords(lb1gt_1,175,198,444).  visible(lb1gt_1,444). happensAt(walking(lb1gt_1),444). orientation(lb1gt_2,146,444). coords(lb1gt_2,187,148,444).  visible(lb1gt_2,444). happensAt(walking(lb1gt_2),444).
orientation(lb1gt_1,1,445). coords(lb1gt_1,176,199,445).  visible(lb1gt_1,445). happensAt(walking(lb1gt_1),445). orientation(lb1gt_2,146,445). coords(lb1gt_2,187,150,445).  visible(lb1gt_2,445). happensAt(walking(lb1gt_2),445).
orientation(lb1gt_1,2,446). coords(lb1gt_1,176,199,446).  visible(lb1gt_1,446). happensAt(walking(lb1gt_1),446). orientation(lb1gt_2,146,446). coords(lb1gt_2,187,151,446).  visible(lb1gt_2,446). happensAt(walking(lb1gt_2),446).
orientation(lb1gt_1,4,447). coords(lb1gt_1,177,200,447).  visible(lb1gt_1,447). happensAt(walking(lb1gt_1),447). orientation(lb1gt_2,146,447). coords(lb1gt_2,187,153,447).  visible(lb1gt_2,447). happensAt(walking(lb1gt_2),447).
orientation(lb1gt_1,0,448). coords(lb1gt_1,178,201,448).  visible(lb1gt_1,448). happensAt(walking(lb1gt_1),448). orientation(lb1gt_2,146,448). coords(lb1gt_2,186,154,448).  visible(lb1gt_2,448). happensAt(walking(lb1gt_2),448).
orientation(lb1gt_1,0,449). coords(lb1gt_1,178,202,449).  visible(lb1gt_1,449). happensAt(walking(lb1gt_1),449). orientation(lb1gt_2,146,449). coords(lb1gt_2,187,155,449).  visible(lb1gt_2,449). happensAt(walking(lb1gt_2),449).
orientation(lb1gt_1,172,450). coords(lb1gt_1,178,204,450).  visible(lb1gt_1,450). happensAt(walking(lb1gt_1),450). orientation(lb1gt_2,146,450). coords(lb1gt_2,187,156,450).  visible(lb1gt_2,450). happensAt(walking(lb1gt_2),450).
orientation(lb1gt_1,175,451). coords(lb1gt_1,178,206,451).  visible(lb1gt_1,451). happensAt(walking(lb1gt_1),451). orientation(lb1gt_2,146,451). coords(lb1gt_2,187,156,451).  visible(lb1gt_2,451). happensAt(walking(lb1gt_2),451).
orientation(lb1gt_1,0,452). coords(lb1gt_1,179,208,452).  visible(lb1gt_1,452). happensAt(walking(lb1gt_1),452). orientation(lb1gt_2,146,452). coords(lb1gt_2,187,157,452).  visible(lb1gt_2,452). happensAt(walking(lb1gt_2),452).
orientation(lb1gt_1,0,453). coords(lb1gt_1,179,209,453).  visible(lb1gt_1,453). happensAt(walking(lb1gt_1),453). orientation(lb1gt_2,146,453). coords(lb1gt_2,187,158,453).  visible(lb1gt_2,453). happensAt(walking(lb1gt_2),453).
orientation(lb1gt_1,178,454). coords(lb1gt_1,179,211,454).  visible(lb1gt_1,454). happensAt(walking(lb1gt_1),454). orientation(lb1gt_2,146,454). coords(lb1gt_2,187,159,454).  visible(lb1gt_2,454). happensAt(walking(lb1gt_2),454).
orientation(lb1gt_1,178,455). coords(lb1gt_1,180,213,455).  visible(lb1gt_1,455). happensAt(walking(lb1gt_1),455). orientation(lb1gt_2,146,455). coords(lb1gt_2,187,159,455).  visible(lb1gt_2,455). happensAt(walking(lb1gt_2),455).
orientation(lb1gt_1,176,456). coords(lb1gt_1,181,217,456).  visible(lb1gt_1,456). happensAt(walking(lb1gt_1),456). orientation(lb1gt_2,146,456). coords(lb1gt_2,187,160,456).  visible(lb1gt_2,456). happensAt(walking(lb1gt_2),456).
orientation(lb1gt_1,176,457). coords(lb1gt_1,183,220,457).  visible(lb1gt_1,457). happensAt(walking(lb1gt_1),457). orientation(lb1gt_2,146,457). coords(lb1gt_2,187,160,457).  visible(lb1gt_2,457). happensAt(walking(lb1gt_2),457).
orientation(lb1gt_1,176,458). coords(lb1gt_1,184,221,458).  visible(lb1gt_1,458). happensAt(walking(lb1gt_1),458). orientation(lb1gt_2,146,458). coords(lb1gt_2,187,161,458).  visible(lb1gt_2,458). happensAt(walking(lb1gt_2),458).
orientation(lb1gt_1,176,459). coords(lb1gt_1,185,222,459).  visible(lb1gt_1,459). happensAt(walking(lb1gt_1),459). orientation(lb1gt_2,146,459). coords(lb1gt_2,188,161,459).  visible(lb1gt_2,459). happensAt(walking(lb1gt_2),459).
orientation(lb1gt_1,176,460). coords(lb1gt_1,186,222,460).  visible(lb1gt_1,460). happensAt(walking(lb1gt_1),460). orientation(lb1gt_2,146,460). coords(lb1gt_2,188,162,460).  visible(lb1gt_2,460). happensAt(walking(lb1gt_2),460).
orientation(lb1gt_1,176,461). coords(lb1gt_1,187,223,461).  visible(lb1gt_1,461). happensAt(walking(lb1gt_1),461). orientation(lb1gt_2,146,461). coords(lb1gt_2,188,163,461).  visible(lb1gt_2,461). happensAt(walking(lb1gt_2),461).
orientation(lb1gt_1,176,462). coords(lb1gt_1,187,223,462).  visible(lb1gt_1,462). happensAt(walking(lb1gt_1),462). orientation(lb1gt_2,146,462). coords(lb1gt_2,188,164,462).  visible(lb1gt_2,462). happensAt(walking(lb1gt_2),462).
orientation(lb1gt_1,176,463). coords(lb1gt_1,188,224,463).  visible(lb1gt_1,463). happensAt(walking(lb1gt_1),463). orientation(lb1gt_2,146,463). coords(lb1gt_2,188,165,463).  visible(lb1gt_2,463). happensAt(walking(lb1gt_2),463).
orientation(lb1gt_1,2,464). coords(lb1gt_1,188,225,464).  visible(lb1gt_1,464). happensAt(walking(lb1gt_1),464). orientation(lb1gt_2,146,464). coords(lb1gt_2,188,168,464).  visible(lb1gt_2,464). happensAt(walking(lb1gt_2),464).
orientation(lb1gt_1,12,465). coords(lb1gt_1,188,226,465).  visible(lb1gt_1,465). happensAt(walking(lb1gt_1),465). orientation(lb1gt_2,146,465). coords(lb1gt_2,188,171,465).  visible(lb1gt_2,465). happensAt(walking(lb1gt_2),465).
orientation(lb1gt_1,12,466). coords(lb1gt_1,189,230,466).  visible(lb1gt_1,466). happensAt(walking(lb1gt_1),466). orientation(lb1gt_2,146,466). coords(lb1gt_2,188,173,466).  visible(lb1gt_2,466). happensAt(walking(lb1gt_2),466).
orientation(lb1gt_1,12,467). coords(lb1gt_1,189,232,467).  visible(lb1gt_1,467). happensAt(walking(lb1gt_1),467). orientation(lb1gt_2,146,467). coords(lb1gt_2,188,175,467).  visible(lb1gt_2,467). happensAt(walking(lb1gt_2),467).
orientation(lb1gt_1,12,468). coords(lb1gt_1,189,233,468).  visible(lb1gt_1,468). happensAt(walking(lb1gt_1),468). orientation(lb1gt_2,146,468). coords(lb1gt_2,188,177,468).  visible(lb1gt_2,468). happensAt(walking(lb1gt_2),468).
#show holdsAt/2.