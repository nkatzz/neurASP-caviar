
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

	orientation(ms3ggt_0,160,434). coords(ms3ggt_0,109,144,434).  visible(ms3ggt_0,434). happensAt(walking(ms3ggt_0),434). orientation(ms3ggt_1,158,434). coords(ms3ggt_1,188,161,434).  visible(ms3ggt_1,434). happensAt(walking(ms3ggt_1),434).
orientation(ms3ggt_0,160,435). coords(ms3ggt_0,108,144,435).  visible(ms3ggt_0,435). happensAt(walking(ms3ggt_0),435). orientation(ms3ggt_1,158,435). coords(ms3ggt_1,188,161,435).  visible(ms3ggt_1,435). happensAt(walking(ms3ggt_1),435).
orientation(ms3ggt_0,160,436). coords(ms3ggt_0,107,145,436).  visible(ms3ggt_0,436). happensAt(walking(ms3ggt_0),436). orientation(ms3ggt_1,158,436). coords(ms3ggt_1,189,161,436).  visible(ms3ggt_1,436). happensAt(walking(ms3ggt_1),436).
orientation(ms3ggt_0,160,437). coords(ms3ggt_0,107,146,437).  visible(ms3ggt_0,437). happensAt(walking(ms3ggt_0),437). orientation(ms3ggt_1,158,437). coords(ms3ggt_1,189,161,437).  visible(ms3ggt_1,437). happensAt(walking(ms3ggt_1),437).
orientation(ms3ggt_0,160,438). coords(ms3ggt_0,107,146,438).  visible(ms3ggt_0,438). happensAt(walking(ms3ggt_0),438). orientation(ms3ggt_1,158,438). coords(ms3ggt_1,191,161,438).  visible(ms3ggt_1,438). happensAt(walking(ms3ggt_1),438).
orientation(ms3ggt_0,160,439). coords(ms3ggt_0,107,146,439).  visible(ms3ggt_0,439). happensAt(walking(ms3ggt_0),439). orientation(ms3ggt_1,158,439). coords(ms3ggt_1,191,161,439).  visible(ms3ggt_1,439). happensAt(walking(ms3ggt_1),439).
orientation(ms3ggt_0,168,440). coords(ms3ggt_0,106,146,440).  visible(ms3ggt_0,440). happensAt(walking(ms3ggt_0),440). orientation(ms3ggt_1,158,440). coords(ms3ggt_1,192,161,440).  visible(ms3ggt_1,440). happensAt(walking(ms3ggt_1),440).
orientation(ms3ggt_0,168,441). coords(ms3ggt_0,105,147,441).  visible(ms3ggt_0,441). happensAt(walking(ms3ggt_0),441). orientation(ms3ggt_1,158,441). coords(ms3ggt_1,193,163,441).  visible(ms3ggt_1,441). happensAt(walking(ms3ggt_1),441).
orientation(ms3ggt_0,168,442). coords(ms3ggt_0,105,147,442).  visible(ms3ggt_0,442). happensAt(walking(ms3ggt_0),442). orientation(ms3ggt_1,167,442). coords(ms3ggt_1,196,164,442).  visible(ms3ggt_1,442). happensAt(walking(ms3ggt_1),442).
orientation(ms3ggt_0,170,443). coords(ms3ggt_0,103,147,443).  visible(ms3ggt_0,443). happensAt(walking(ms3ggt_0),443). orientation(ms3ggt_1,171,443). coords(ms3ggt_1,199,165,443).  visible(ms3ggt_1,443). happensAt(walking(ms3ggt_1),443).
orientation(ms3ggt_0,171,444). coords(ms3ggt_0,101,148,444).  visible(ms3ggt_0,444). happensAt(walking(ms3ggt_0),444). orientation(ms3ggt_1,171,444). coords(ms3ggt_1,202,165,444).  visible(ms3ggt_1,444). happensAt(walking(ms3ggt_1),444).
orientation(ms3ggt_0,171,445). coords(ms3ggt_0,100,148,445).  visible(ms3ggt_0,445). happensAt(walking(ms3ggt_0),445). orientation(ms3ggt_1,177,445). coords(ms3ggt_1,206,166,445).  visible(ms3ggt_1,445). happensAt(walking(ms3ggt_1),445).
orientation(ms3ggt_0,171,446). coords(ms3ggt_0,98,148,446).  visible(ms3ggt_0,446). happensAt(walking(ms3ggt_0),446). orientation(ms3ggt_1,177,446). coords(ms3ggt_1,208,166,446).  visible(ms3ggt_1,446). happensAt(walking(ms3ggt_1),446).
orientation(ms3ggt_0,171,447). coords(ms3ggt_0,97,148,447).  visible(ms3ggt_0,447). happensAt(walking(ms3ggt_0),447). orientation(ms3ggt_1,177,447). coords(ms3ggt_1,209,167,447).  visible(ms3ggt_1,447). happensAt(walking(ms3ggt_1),447).
orientation(ms3ggt_0,171,448). coords(ms3ggt_0,96,148,448).  visible(ms3ggt_0,448). happensAt(walking(ms3ggt_0),448). orientation(ms3ggt_1,177,448). coords(ms3ggt_1,212,168,448).  visible(ms3ggt_1,448). happensAt(walking(ms3ggt_1),448).
orientation(ms3ggt_0,171,449). coords(ms3ggt_0,96,148,449).  visible(ms3ggt_0,449). happensAt(walking(ms3ggt_0),449). orientation(ms3ggt_1,177,449). coords(ms3ggt_1,213,169,449).  visible(ms3ggt_1,449). happensAt(walking(ms3ggt_1),449).
orientation(ms3ggt_0,166,450). coords(ms3ggt_0,87,149,450).  visible(ms3ggt_0,450). happensAt(walking(ms3ggt_0),450). orientation(ms3ggt_1,177,450). coords(ms3ggt_1,214,170,450).  visible(ms3ggt_1,450). happensAt(walking(ms3ggt_1),450).
orientation(ms3ggt_0,166,451). coords(ms3ggt_0,87,149,451).  visible(ms3ggt_0,451). happensAt(walking(ms3ggt_0),451). orientation(ms3ggt_1,177,451). coords(ms3ggt_1,214,171,451).  visible(ms3ggt_1,451). happensAt(walking(ms3ggt_1),451).
orientation(ms3ggt_0,166,452). coords(ms3ggt_0,85,150,452).  visible(ms3ggt_0,452). happensAt(walking(ms3ggt_0),452). orientation(ms3ggt_1,177,452). coords(ms3ggt_1,215,173,452).  visible(ms3ggt_1,452). happensAt(walking(ms3ggt_1),452).
orientation(ms3ggt_0,166,453). coords(ms3ggt_0,85,150,453).  visible(ms3ggt_0,453). happensAt(walking(ms3ggt_0),453). orientation(ms3ggt_1,177,453). coords(ms3ggt_1,215,175,453).  visible(ms3ggt_1,453). happensAt(walking(ms3ggt_1),453).
orientation(ms3ggt_0,166,454). coords(ms3ggt_0,83,151,454).  visible(ms3ggt_0,454). happensAt(walking(ms3ggt_0),454). orientation(ms3ggt_1,177,454). coords(ms3ggt_1,216,177,454).  visible(ms3ggt_1,454). happensAt(walking(ms3ggt_1),454).
orientation(ms3ggt_0,166,455). coords(ms3ggt_0,82,151,455).  visible(ms3ggt_0,455). happensAt(walking(ms3ggt_0),455). orientation(ms3ggt_1,177,455). coords(ms3ggt_1,217,179,455).  visible(ms3ggt_1,455). happensAt(walking(ms3ggt_1),455).
orientation(ms3ggt_0,166,456). coords(ms3ggt_0,82,151,456).  visible(ms3ggt_0,456). happensAt(walking(ms3ggt_0),456). orientation(ms3ggt_1,177,456). coords(ms3ggt_1,218,181,456).  visible(ms3ggt_1,456). happensAt(walking(ms3ggt_1),456).
orientation(ms3ggt_0,166,457). coords(ms3ggt_0,78,152,457).  visible(ms3ggt_0,457). happensAt(walking(ms3ggt_0),457). orientation(ms3ggt_1,177,457). coords(ms3ggt_1,218,185,457).  visible(ms3ggt_1,457). happensAt(walking(ms3ggt_1),457).
orientation(ms3ggt_0,166,458). coords(ms3ggt_0,77,152,458).  visible(ms3ggt_0,458). happensAt(walking(ms3ggt_0),458). orientation(ms3ggt_1,177,458). coords(ms3ggt_1,219,189,458).  visible(ms3ggt_1,458). happensAt(walking(ms3ggt_1),458).
#show holdsAt/2.