
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

	orientation(ms3ggt_1,158,440). coords(ms3ggt_1,192,161,440).  visible(ms3ggt_1,440). happensAt(walking(ms3ggt_1),440). orientation(ms3ggt_2,155,440). coords(ms3ggt_2,110,169,440).  visible(ms3ggt_2,440). happensAt(walking(ms3ggt_2),440).
orientation(ms3ggt_1,158,441). coords(ms3ggt_1,193,163,441).  visible(ms3ggt_1,441). happensAt(walking(ms3ggt_1),441). orientation(ms3ggt_2,155,441). coords(ms3ggt_2,109,167,441).  visible(ms3ggt_2,441). happensAt(walking(ms3ggt_2),441).
orientation(ms3ggt_1,167,442). coords(ms3ggt_1,196,164,442).  visible(ms3ggt_1,442). happensAt(walking(ms3ggt_1),442). orientation(ms3ggt_2,155,442). coords(ms3ggt_2,109,167,442).  visible(ms3ggt_2,442). happensAt(walking(ms3ggt_2),442).
orientation(ms3ggt_1,171,443). coords(ms3ggt_1,199,165,443).  visible(ms3ggt_1,443). happensAt(walking(ms3ggt_1),443). orientation(ms3ggt_2,155,443). coords(ms3ggt_2,109,167,443).  visible(ms3ggt_2,443). happensAt(walking(ms3ggt_2),443).
orientation(ms3ggt_1,171,444). coords(ms3ggt_1,202,165,444).  visible(ms3ggt_1,444). happensAt(walking(ms3ggt_1),444). orientation(ms3ggt_2,155,444). coords(ms3ggt_2,109,164,444).  visible(ms3ggt_2,444). happensAt(walking(ms3ggt_2),444).
orientation(ms3ggt_1,177,445). coords(ms3ggt_1,206,166,445).  visible(ms3ggt_1,445). happensAt(walking(ms3ggt_1),445). orientation(ms3ggt_2,155,445). coords(ms3ggt_2,109,164,445).  visible(ms3ggt_2,445). happensAt(walking(ms3ggt_2),445).
orientation(ms3ggt_1,177,446). coords(ms3ggt_1,208,166,446).  visible(ms3ggt_1,446). happensAt(walking(ms3ggt_1),446). orientation(ms3ggt_2,155,446). coords(ms3ggt_2,108,164,446).  visible(ms3ggt_2,446). happensAt(walking(ms3ggt_2),446).
orientation(ms3ggt_1,177,447). coords(ms3ggt_1,209,167,447).  visible(ms3ggt_1,447). happensAt(walking(ms3ggt_1),447). orientation(ms3ggt_2,155,447). coords(ms3ggt_2,108,164,447).  visible(ms3ggt_2,447). happensAt(walking(ms3ggt_2),447).
orientation(ms3ggt_1,177,448). coords(ms3ggt_1,212,168,448).  visible(ms3ggt_1,448). happensAt(walking(ms3ggt_1),448). orientation(ms3ggt_2,155,448). coords(ms3ggt_2,108,163,448).  visible(ms3ggt_2,448). happensAt(walking(ms3ggt_2),448).
orientation(ms3ggt_1,177,449). coords(ms3ggt_1,213,169,449).  visible(ms3ggt_1,449). happensAt(walking(ms3ggt_1),449). orientation(ms3ggt_2,155,449). coords(ms3ggt_2,108,162,449).  visible(ms3ggt_2,449). happensAt(walking(ms3ggt_2),449).
orientation(ms3ggt_1,177,450). coords(ms3ggt_1,214,170,450).  visible(ms3ggt_1,450). happensAt(walking(ms3ggt_1),450). orientation(ms3ggt_2,155,450). coords(ms3ggt_2,108,161,450).  visible(ms3ggt_2,450). happensAt(walking(ms3ggt_2),450).
orientation(ms3ggt_1,177,451). coords(ms3ggt_1,214,171,451).  visible(ms3ggt_1,451). happensAt(walking(ms3ggt_1),451). orientation(ms3ggt_2,155,451). coords(ms3ggt_2,107,161,451).  visible(ms3ggt_2,451). happensAt(walking(ms3ggt_2),451).
orientation(ms3ggt_1,177,452). coords(ms3ggt_1,215,173,452).  visible(ms3ggt_1,452). happensAt(walking(ms3ggt_1),452). orientation(ms3ggt_2,155,452). coords(ms3ggt_2,106,160,452).  visible(ms3ggt_2,452). happensAt(walking(ms3ggt_2),452).
orientation(ms3ggt_1,177,453). coords(ms3ggt_1,215,175,453).  visible(ms3ggt_1,453). happensAt(walking(ms3ggt_1),453). orientation(ms3ggt_2,156,453). coords(ms3ggt_2,106,159,453).  visible(ms3ggt_2,453). happensAt(walking(ms3ggt_2),453).
orientation(ms3ggt_1,177,454). coords(ms3ggt_1,216,177,454).  visible(ms3ggt_1,454). happensAt(walking(ms3ggt_1),454). orientation(ms3ggt_2,156,454). coords(ms3ggt_2,105,159,454).  visible(ms3ggt_2,454). happensAt(walking(ms3ggt_2),454).
orientation(ms3ggt_1,177,455). coords(ms3ggt_1,217,179,455).  visible(ms3ggt_1,455). happensAt(walking(ms3ggt_1),455). orientation(ms3ggt_2,156,455). coords(ms3ggt_2,105,158,455).  visible(ms3ggt_2,455). happensAt(walking(ms3ggt_2),455).
orientation(ms3ggt_1,177,456). coords(ms3ggt_1,218,181,456).  visible(ms3ggt_1,456). happensAt(walking(ms3ggt_1),456). orientation(ms3ggt_2,156,456). coords(ms3ggt_2,104,157,456).  visible(ms3ggt_2,456). happensAt(walking(ms3ggt_2),456).
orientation(ms3ggt_1,177,457). coords(ms3ggt_1,218,185,457).  visible(ms3ggt_1,457). happensAt(walking(ms3ggt_1),457). orientation(ms3ggt_2,156,457). coords(ms3ggt_2,104,158,457).  visible(ms3ggt_2,457). happensAt(walking(ms3ggt_2),457).
orientation(ms3ggt_1,177,458). coords(ms3ggt_1,219,189,458).  visible(ms3ggt_1,458). happensAt(walking(ms3ggt_1),458). orientation(ms3ggt_2,156,458). coords(ms3ggt_2,103,156,458).  visible(ms3ggt_2,458). happensAt(walking(ms3ggt_2),458).
orientation(ms3ggt_1,177,459). coords(ms3ggt_1,221,192,459).  visible(ms3ggt_1,459). happensAt(walking(ms3ggt_1),459). orientation(ms3ggt_2,156,459). coords(ms3ggt_2,103,156,459).  visible(ms3ggt_2,459). happensAt(walking(ms3ggt_2),459).
orientation(ms3ggt_1,170,460). coords(ms3ggt_1,221,193,460).  visible(ms3ggt_1,460). happensAt(walking(ms3ggt_1),460). orientation(ms3ggt_2,156,460). coords(ms3ggt_2,103,155,460).  visible(ms3ggt_2,460). happensAt(walking(ms3ggt_2),460).
orientation(ms3ggt_1,170,461). coords(ms3ggt_1,224,197,461).  visible(ms3ggt_1,461). happensAt(walking(ms3ggt_1),461). orientation(ms3ggt_2,156,461). coords(ms3ggt_2,103,155,461).  visible(ms3ggt_2,461). happensAt(walking(ms3ggt_2),461).
orientation(ms3ggt_1,170,462). coords(ms3ggt_1,225,198,462).  visible(ms3ggt_1,462). happensAt(walking(ms3ggt_1),462). orientation(ms3ggt_2,151,462). coords(ms3ggt_2,103,154,462).  visible(ms3ggt_2,462). happensAt(walking(ms3ggt_2),462).
orientation(ms3ggt_1,170,463). coords(ms3ggt_1,226,198,463).  visible(ms3ggt_1,463). happensAt(walking(ms3ggt_1),463). orientation(ms3ggt_2,151,463). coords(ms3ggt_2,103,153,463).  visible(ms3ggt_2,463). happensAt(walking(ms3ggt_2),463).
orientation(ms3ggt_1,170,464). coords(ms3ggt_1,226,198,464).  visible(ms3ggt_1,464). happensAt(walking(ms3ggt_1),464). orientation(ms3ggt_2,151,464). coords(ms3ggt_2,102,152,464).  visible(ms3ggt_2,464). happensAt(walking(ms3ggt_2),464).
#show holdsAt/2.