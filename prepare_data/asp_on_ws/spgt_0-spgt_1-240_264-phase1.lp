
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

	orientation(spgt_0,148,240). coords(spgt_0,116,112,240).  visible(spgt_0,240). happensAt(active(spgt_0),240). orientation(spgt_1,141,240). coords(spgt_1,115,96,240).  visible(spgt_1,240). happensAt(walking(spgt_1),240).
orientation(spgt_0,148,241). coords(spgt_0,117,113,241).  visible(spgt_0,241). happensAt(active(spgt_0),241). orientation(spgt_1,141,241). coords(spgt_1,116,96,241).  visible(spgt_1,241). happensAt(walking(spgt_1),241).
orientation(spgt_0,148,242). coords(spgt_0,118,112,242).  visible(spgt_0,242). happensAt(active(spgt_0),242). orientation(spgt_1,141,242). coords(spgt_1,116,96,242).  visible(spgt_1,242). happensAt(walking(spgt_1),242).
orientation(spgt_0,148,243). coords(spgt_0,118,112,243).  visible(spgt_0,243). happensAt(active(spgt_0),243). orientation(spgt_1,141,243). coords(spgt_1,116,95,243).  visible(spgt_1,243). happensAt(walking(spgt_1),243).
orientation(spgt_0,148,244). coords(spgt_0,118,112,244).  visible(spgt_0,244). happensAt(active(spgt_0),244). orientation(spgt_1,141,244). coords(spgt_1,116,95,244).  visible(spgt_1,244). happensAt(walking(spgt_1),244).
orientation(spgt_0,148,245). coords(spgt_0,118,111,245).  visible(spgt_0,245). happensAt(active(spgt_0),245). orientation(spgt_1,137,245). coords(spgt_1,117,94,245).  visible(spgt_1,245). happensAt(walking(spgt_1),245).
orientation(spgt_0,148,246). coords(spgt_0,119,111,246).  visible(spgt_0,246). happensAt(active(spgt_0),246). orientation(spgt_1,137,246). coords(spgt_1,117,93,246).  visible(spgt_1,246). happensAt(walking(spgt_1),246).
orientation(spgt_0,148,247). coords(spgt_0,119,112,247).  visible(spgt_0,247). happensAt(active(spgt_0),247). orientation(spgt_1,137,247). coords(spgt_1,117,91,247).  visible(spgt_1,247). happensAt(walking(spgt_1),247).
orientation(spgt_0,148,248). coords(spgt_0,119,111,248).  visible(spgt_0,248). happensAt(active(spgt_0),248). orientation(spgt_1,137,248). coords(spgt_1,117,90,248).  visible(spgt_1,248). happensAt(walking(spgt_1),248).
orientation(spgt_0,148,249). coords(spgt_0,119,111,249).  visible(spgt_0,249). happensAt(active(spgt_0),249). orientation(spgt_1,137,249). coords(spgt_1,118,89,249).  visible(spgt_1,249). happensAt(walking(spgt_1),249).
orientation(spgt_0,148,250). coords(spgt_0,120,111,250).  visible(spgt_0,250). happensAt(walking(spgt_0),250). orientation(spgt_1,137,250). coords(spgt_1,118,87,250).  visible(spgt_1,250). happensAt(walking(spgt_1),250).
orientation(spgt_0,148,251). coords(spgt_0,120,110,251).  visible(spgt_0,251). happensAt(walking(spgt_0),251). orientation(spgt_1,137,251). coords(spgt_1,118,87,251).  visible(spgt_1,251). happensAt(walking(spgt_1),251).
orientation(spgt_0,148,252). coords(spgt_0,123,110,252).  visible(spgt_0,252). happensAt(walking(spgt_0),252). orientation(spgt_1,137,252). coords(spgt_1,119,86,252).  visible(spgt_1,252). happensAt(walking(spgt_1),252).
orientation(spgt_0,148,253). coords(spgt_0,124,110,253).  visible(spgt_0,253). happensAt(walking(spgt_0),253). orientation(spgt_1,137,253). coords(spgt_1,120,85,253).  visible(spgt_1,253). happensAt(walking(spgt_1),253).
orientation(spgt_0,148,254). coords(spgt_0,125,110,254).  visible(spgt_0,254). happensAt(walking(spgt_0),254). orientation(spgt_1,137,254). coords(spgt_1,120,85,254).  visible(spgt_1,254). happensAt(walking(spgt_1),254).
orientation(spgt_0,148,255). coords(spgt_0,127,110,255).  visible(spgt_0,255). happensAt(walking(spgt_0),255). orientation(spgt_1,137,255). coords(spgt_1,121,85,255).  visible(spgt_1,255). happensAt(walking(spgt_1),255).
orientation(spgt_0,148,256). coords(spgt_0,128,110,256).  visible(spgt_0,256). happensAt(walking(spgt_0),256). orientation(spgt_1,137,256). coords(spgt_1,121,84,256).  visible(spgt_1,256). happensAt(walking(spgt_1),256).
orientation(spgt_0,148,257). coords(spgt_0,129,109,257).  visible(spgt_0,257). happensAt(walking(spgt_0),257). orientation(spgt_1,137,257). coords(spgt_1,122,83,257).  visible(spgt_1,257). happensAt(walking(spgt_1),257).
orientation(spgt_0,148,258). coords(spgt_0,130,109,258).  visible(spgt_0,258). happensAt(walking(spgt_0),258). orientation(spgt_1,137,258). coords(spgt_1,122,82,258).  visible(spgt_1,258). happensAt(walking(spgt_1),258).
orientation(spgt_0,148,259). coords(spgt_0,130,108,259).  visible(spgt_0,259). happensAt(walking(spgt_0),259). orientation(spgt_1,137,259). coords(spgt_1,122,81,259).  visible(spgt_1,259). happensAt(walking(spgt_1),259).
orientation(spgt_0,148,260). coords(spgt_0,131,108,260).  visible(spgt_0,260). happensAt(walking(spgt_0),260). orientation(spgt_1,137,260). coords(spgt_1,122,80,260).  visible(spgt_1,260). happensAt(walking(spgt_1),260).
orientation(spgt_0,148,261). coords(spgt_0,131,107,261).  visible(spgt_0,261). happensAt(walking(spgt_0),261). orientation(spgt_1,137,261). coords(spgt_1,122,78,261).  visible(spgt_1,261). happensAt(walking(spgt_1),261).
orientation(spgt_0,148,262). coords(spgt_0,132,107,262).  visible(spgt_0,262). happensAt(walking(spgt_0),262). orientation(spgt_1,137,262). coords(spgt_1,123,77,262).  visible(spgt_1,262). happensAt(walking(spgt_1),262).
orientation(spgt_0,148,263). coords(spgt_0,132,106,263).  visible(spgt_0,263). happensAt(walking(spgt_0),263). orientation(spgt_1,137,263). coords(spgt_1,123,75,263).  visible(spgt_1,263). happensAt(walking(spgt_1),263).
orientation(spgt_0,148,264). coords(spgt_0,132,105,264).  visible(spgt_0,264). happensAt(walking(spgt_0),264). orientation(spgt_1,137,264). coords(spgt_1,123,73,264).  visible(spgt_1,264). happensAt(walking(spgt_1),264).
#show holdsAt/2.