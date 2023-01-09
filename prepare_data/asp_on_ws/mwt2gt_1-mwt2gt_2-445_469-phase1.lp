
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

	orientation(mwt2gt_1,155,445). coords(mwt2gt_1,81,103,445).  visible(mwt2gt_1,445). happensAt(walking(mwt2gt_1),445). orientation(mwt2gt_2,150,445). coords(mwt2gt_2,80,128,445).  visible(mwt2gt_2,445). happensAt(walking(mwt2gt_2),445).
orientation(mwt2gt_1,155,446). coords(mwt2gt_1,80,104,446).  visible(mwt2gt_1,446). happensAt(walking(mwt2gt_1),446). orientation(mwt2gt_2,150,446). coords(mwt2gt_2,79,129,446).  visible(mwt2gt_2,446). happensAt(walking(mwt2gt_2),446).
orientation(mwt2gt_1,155,447). coords(mwt2gt_1,78,104,447).  visible(mwt2gt_1,447). happensAt(walking(mwt2gt_1),447). orientation(mwt2gt_2,150,447). coords(mwt2gt_2,78,129,447).  visible(mwt2gt_2,447). happensAt(walking(mwt2gt_2),447).
orientation(mwt2gt_1,155,448). coords(mwt2gt_1,78,104,448).  visible(mwt2gt_1,448). happensAt(walking(mwt2gt_1),448). orientation(mwt2gt_2,150,448). coords(mwt2gt_2,78,130,448).  visible(mwt2gt_2,448). happensAt(walking(mwt2gt_2),448).
orientation(mwt2gt_1,155,449). coords(mwt2gt_1,78,105,449).  visible(mwt2gt_1,449). happensAt(walking(mwt2gt_1),449). orientation(mwt2gt_2,150,449). coords(mwt2gt_2,77,130,449).  visible(mwt2gt_2,449). happensAt(walking(mwt2gt_2),449).
orientation(mwt2gt_1,155,450). coords(mwt2gt_1,77,106,450).  visible(mwt2gt_1,450). happensAt(walking(mwt2gt_1),450). orientation(mwt2gt_2,150,450). coords(mwt2gt_2,75,130,450).  visible(mwt2gt_2,450). happensAt(walking(mwt2gt_2),450).
orientation(mwt2gt_1,155,451). coords(mwt2gt_1,77,109,451).  visible(mwt2gt_1,451). happensAt(walking(mwt2gt_1),451). orientation(mwt2gt_2,157,451). coords(mwt2gt_2,74,131,451).  visible(mwt2gt_2,451). happensAt(walking(mwt2gt_2),451).
orientation(mwt2gt_1,155,452). coords(mwt2gt_1,77,110,452).  visible(mwt2gt_1,452). happensAt(walking(mwt2gt_1),452). orientation(mwt2gt_2,157,452). coords(mwt2gt_2,74,131,452).  visible(mwt2gt_2,452). happensAt(walking(mwt2gt_2),452).
orientation(mwt2gt_1,155,453). coords(mwt2gt_1,77,111,453).  visible(mwt2gt_1,453). happensAt(walking(mwt2gt_1),453). orientation(mwt2gt_2,157,453). coords(mwt2gt_2,73,131,453).  visible(mwt2gt_2,453). happensAt(walking(mwt2gt_2),453).
orientation(mwt2gt_1,155,454). coords(mwt2gt_1,77,113,454).  visible(mwt2gt_1,454). happensAt(walking(mwt2gt_1),454). orientation(mwt2gt_2,157,454). coords(mwt2gt_2,73,132,454).  visible(mwt2gt_2,454). happensAt(walking(mwt2gt_2),454).
orientation(mwt2gt_1,155,455). coords(mwt2gt_1,76,114,455).  visible(mwt2gt_1,455). happensAt(walking(mwt2gt_1),455). orientation(mwt2gt_2,157,455). coords(mwt2gt_2,72,133,455).  visible(mwt2gt_2,455). happensAt(walking(mwt2gt_2),455).
orientation(mwt2gt_1,155,456). coords(mwt2gt_1,75,114,456).  visible(mwt2gt_1,456). happensAt(walking(mwt2gt_1),456). orientation(mwt2gt_2,157,456). coords(mwt2gt_2,72,135,456).  visible(mwt2gt_2,456). happensAt(walking(mwt2gt_2),456).
orientation(mwt2gt_1,155,457). coords(mwt2gt_1,74,115,457).  visible(mwt2gt_1,457). happensAt(walking(mwt2gt_1),457). orientation(mwt2gt_2,157,457). coords(mwt2gt_2,72,135,457).  visible(mwt2gt_2,457). happensAt(walking(mwt2gt_2),457).
orientation(mwt2gt_1,155,458). coords(mwt2gt_1,73,115,458).  visible(mwt2gt_1,458). happensAt(walking(mwt2gt_1),458). orientation(mwt2gt_2,157,458). coords(mwt2gt_2,71,136,458).  visible(mwt2gt_2,458). happensAt(walking(mwt2gt_2),458).
orientation(mwt2gt_1,155,459). coords(mwt2gt_1,72,116,459).  visible(mwt2gt_1,459). happensAt(walking(mwt2gt_1),459). orientation(mwt2gt_2,157,459). coords(mwt2gt_2,71,137,459).  visible(mwt2gt_2,459). happensAt(walking(mwt2gt_2),459).
orientation(mwt2gt_1,155,460). coords(mwt2gt_1,71,116,460).  visible(mwt2gt_1,460). happensAt(walking(mwt2gt_1),460). orientation(mwt2gt_2,157,460). coords(mwt2gt_2,71,137,460).  visible(mwt2gt_2,460). happensAt(walking(mwt2gt_2),460).
orientation(mwt2gt_1,155,461). coords(mwt2gt_1,70,118,461).  visible(mwt2gt_1,461). happensAt(walking(mwt2gt_1),461). orientation(mwt2gt_2,157,461). coords(mwt2gt_2,70,138,461).  visible(mwt2gt_2,461). happensAt(walking(mwt2gt_2),461).
orientation(mwt2gt_1,155,462). coords(mwt2gt_1,70,120,462).  visible(mwt2gt_1,462). happensAt(walking(mwt2gt_1),462). orientation(mwt2gt_2,157,462). coords(mwt2gt_2,69,138,462).  visible(mwt2gt_2,462). happensAt(walking(mwt2gt_2),462).
orientation(mwt2gt_1,155,463). coords(mwt2gt_1,69,122,463).  visible(mwt2gt_1,463). happensAt(walking(mwt2gt_1),463). orientation(mwt2gt_2,157,463). coords(mwt2gt_2,68,138,463).  visible(mwt2gt_2,463). happensAt(walking(mwt2gt_2),463).
orientation(mwt2gt_1,153,464). coords(mwt2gt_1,70,123,464).  visible(mwt2gt_1,464). happensAt(walking(mwt2gt_1),464). orientation(mwt2gt_2,157,464). coords(mwt2gt_2,68,139,464).  visible(mwt2gt_2,464). happensAt(walking(mwt2gt_2),464).
orientation(mwt2gt_1,153,465). coords(mwt2gt_1,69,125,465).  visible(mwt2gt_1,465). happensAt(walking(mwt2gt_1),465). orientation(mwt2gt_2,157,465). coords(mwt2gt_2,66,139,465).  visible(mwt2gt_2,465). happensAt(walking(mwt2gt_2),465).
orientation(mwt2gt_1,160,466). coords(mwt2gt_1,69,126,466).  visible(mwt2gt_1,466). happensAt(walking(mwt2gt_1),466). orientation(mwt2gt_2,157,466). coords(mwt2gt_2,65,141,466).  visible(mwt2gt_2,466). happensAt(walking(mwt2gt_2),466).
orientation(mwt2gt_1,160,467). coords(mwt2gt_1,68,126,467).  visible(mwt2gt_1,467). happensAt(walking(mwt2gt_1),467). orientation(mwt2gt_2,157,467). coords(mwt2gt_2,65,143,467).  visible(mwt2gt_2,467). happensAt(walking(mwt2gt_2),467).
orientation(mwt2gt_1,160,468). coords(mwt2gt_1,68,127,468).  visible(mwt2gt_1,468). happensAt(walking(mwt2gt_1),468). orientation(mwt2gt_2,157,468). coords(mwt2gt_2,65,144,468).  visible(mwt2gt_2,468). happensAt(walking(mwt2gt_2),468).
orientation(mwt2gt_1,164,469). coords(mwt2gt_1,66,128,469).  visible(mwt2gt_1,469). happensAt(walking(mwt2gt_1),469). orientation(mwt2gt_2,157,469). coords(mwt2gt_2,65,146,469).  visible(mwt2gt_2,469). happensAt(walking(mwt2gt_2),469).
#show holdsAt/2.