
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

	orientation(fomdgt3_1,150,240). coords(fomdgt3_1,66,93,240).  visible(fomdgt3_1,240). happensAt(walking(fomdgt3_1),240). orientation(fomdgt3_2,130,240). coords(fomdgt3_2,228,128,240).  visible(fomdgt3_2,240). happensAt(walking(fomdgt3_2),240).
orientation(fomdgt3_1,150,241). coords(fomdgt3_1,64,93,241).  visible(fomdgt3_1,241). happensAt(walking(fomdgt3_1),241). orientation(fomdgt3_2,130,241). coords(fomdgt3_2,228,128,241).  visible(fomdgt3_2,241). happensAt(walking(fomdgt3_2),241).
orientation(fomdgt3_1,150,242). coords(fomdgt3_1,64,93,242).  visible(fomdgt3_1,242). happensAt(walking(fomdgt3_1),242). orientation(fomdgt3_2,130,242). coords(fomdgt3_2,228,128,242).  visible(fomdgt3_2,242). happensAt(walking(fomdgt3_2),242).
orientation(fomdgt3_1,150,243). coords(fomdgt3_1,64,93,243).  visible(fomdgt3_1,243). happensAt(walking(fomdgt3_1),243). orientation(fomdgt3_2,130,243). coords(fomdgt3_2,228,128,243).  visible(fomdgt3_2,243). happensAt(walking(fomdgt3_2),243).
orientation(fomdgt3_1,150,244). coords(fomdgt3_1,64,93,244).  visible(fomdgt3_1,244). happensAt(walking(fomdgt3_1),244). orientation(fomdgt3_2,130,244). coords(fomdgt3_2,228,128,244).  visible(fomdgt3_2,244). happensAt(walking(fomdgt3_2),244).
orientation(fomdgt3_1,150,245). coords(fomdgt3_1,64,93,245).  visible(fomdgt3_1,245). happensAt(walking(fomdgt3_1),245). orientation(fomdgt3_2,130,245). coords(fomdgt3_2,228,128,245).  visible(fomdgt3_2,245). happensAt(walking(fomdgt3_2),245).
orientation(fomdgt3_1,150,246). coords(fomdgt3_1,64,93,246).  visible(fomdgt3_1,246). happensAt(walking(fomdgt3_1),246). orientation(fomdgt3_2,130,246). coords(fomdgt3_2,228,128,246).  visible(fomdgt3_2,246). happensAt(walking(fomdgt3_2),246).
orientation(fomdgt3_1,150,247). coords(fomdgt3_1,64,93,247).  visible(fomdgt3_1,247). happensAt(walking(fomdgt3_1),247). orientation(fomdgt3_2,130,247). coords(fomdgt3_2,228,128,247).  visible(fomdgt3_2,247). happensAt(walking(fomdgt3_2),247).
orientation(fomdgt3_1,150,248). coords(fomdgt3_1,64,93,248).  visible(fomdgt3_1,248). happensAt(walking(fomdgt3_1),248). orientation(fomdgt3_2,130,248). coords(fomdgt3_2,228,128,248).  visible(fomdgt3_2,248). happensAt(walking(fomdgt3_2),248).
orientation(fomdgt3_1,150,249). coords(fomdgt3_1,61,94,249).  visible(fomdgt3_1,249). happensAt(walking(fomdgt3_1),249). orientation(fomdgt3_2,130,249). coords(fomdgt3_2,228,128,249).  visible(fomdgt3_2,249). happensAt(walking(fomdgt3_2),249).
orientation(fomdgt3_1,150,250). coords(fomdgt3_1,61,94,250).  visible(fomdgt3_1,250). happensAt(walking(fomdgt3_1),250). orientation(fomdgt3_2,130,250). coords(fomdgt3_2,228,128,250).  visible(fomdgt3_2,250). happensAt(walking(fomdgt3_2),250).
orientation(fomdgt3_1,150,251). coords(fomdgt3_1,61,94,251).  visible(fomdgt3_1,251). happensAt(walking(fomdgt3_1),251). orientation(fomdgt3_2,130,251). coords(fomdgt3_2,228,128,251).  visible(fomdgt3_2,251). happensAt(walking(fomdgt3_2),251).
orientation(fomdgt3_1,150,252). coords(fomdgt3_1,61,94,252).  visible(fomdgt3_1,252). happensAt(walking(fomdgt3_1),252). orientation(fomdgt3_2,130,252). coords(fomdgt3_2,228,128,252).  visible(fomdgt3_2,252). happensAt(walking(fomdgt3_2),252).
orientation(fomdgt3_1,150,253). coords(fomdgt3_1,61,94,253).  visible(fomdgt3_1,253). happensAt(walking(fomdgt3_1),253). orientation(fomdgt3_2,130,253). coords(fomdgt3_2,228,128,253).  visible(fomdgt3_2,253). happensAt(walking(fomdgt3_2),253).
orientation(fomdgt3_1,150,254). coords(fomdgt3_1,61,94,254).  visible(fomdgt3_1,254). happensAt(walking(fomdgt3_1),254). orientation(fomdgt3_2,130,254). coords(fomdgt3_2,240,139,254).  visible(fomdgt3_2,254). happensAt(walking(fomdgt3_2),254).
orientation(fomdgt3_1,150,255). coords(fomdgt3_1,61,94,255).  visible(fomdgt3_1,255). happensAt(walking(fomdgt3_1),255). orientation(fomdgt3_2,130,255). coords(fomdgt3_2,239,140,255).  visible(fomdgt3_2,255). happensAt(walking(fomdgt3_2),255).
orientation(fomdgt3_1,150,256). coords(fomdgt3_1,61,94,256).  visible(fomdgt3_1,256). happensAt(walking(fomdgt3_1),256). orientation(fomdgt3_2,130,256). coords(fomdgt3_2,239,140,256).  visible(fomdgt3_2,256). happensAt(walking(fomdgt3_2),256).
orientation(fomdgt3_1,150,257). coords(fomdgt3_1,61,94,257).  visible(fomdgt3_1,257). happensAt(walking(fomdgt3_1),257). orientation(fomdgt3_2,130,257). coords(fomdgt3_2,239,140,257).  visible(fomdgt3_2,257). happensAt(walking(fomdgt3_2),257).
orientation(fomdgt3_1,150,258). coords(fomdgt3_1,61,94,258).  visible(fomdgt3_1,258). happensAt(walking(fomdgt3_1),258). orientation(fomdgt3_2,130,258). coords(fomdgt3_2,239,140,258).  visible(fomdgt3_2,258). happensAt(walking(fomdgt3_2),258).
orientation(fomdgt3_1,150,259). coords(fomdgt3_1,61,94,259).  visible(fomdgt3_1,259). happensAt(walking(fomdgt3_1),259). orientation(fomdgt3_2,130,259). coords(fomdgt3_2,239,140,259).  visible(fomdgt3_2,259). happensAt(walking(fomdgt3_2),259).
orientation(fomdgt3_1,150,260). coords(fomdgt3_1,61,94,260).  visible(fomdgt3_1,260). happensAt(walking(fomdgt3_1),260). orientation(fomdgt3_2,130,260). coords(fomdgt3_2,239,140,260).  visible(fomdgt3_2,260). happensAt(walking(fomdgt3_2),260).
orientation(fomdgt3_1,150,261). coords(fomdgt3_1,59,96,261).  visible(fomdgt3_1,261). happensAt(walking(fomdgt3_1),261). orientation(fomdgt3_2,130,261). coords(fomdgt3_2,239,140,261).  visible(fomdgt3_2,261). happensAt(walking(fomdgt3_2),261).
orientation(fomdgt3_1,150,262). coords(fomdgt3_1,59,96,262).  visible(fomdgt3_1,262). happensAt(walking(fomdgt3_1),262). orientation(fomdgt3_2,130,262). coords(fomdgt3_2,239,140,262).  visible(fomdgt3_2,262). happensAt(walking(fomdgt3_2),262).
orientation(fomdgt3_1,150,263). coords(fomdgt3_1,59,96,263).  visible(fomdgt3_1,263). happensAt(walking(fomdgt3_1),263). orientation(fomdgt3_2,130,263). coords(fomdgt3_2,237,140,263).  visible(fomdgt3_2,263). happensAt(walking(fomdgt3_2),263).
orientation(fomdgt3_1,150,264). coords(fomdgt3_1,59,96,264).  visible(fomdgt3_1,264). happensAt(walking(fomdgt3_1),264). orientation(fomdgt3_2,130,264). coords(fomdgt3_2,237,140,264).  visible(fomdgt3_2,264). happensAt(walking(fomdgt3_2),264).
#show holdsAt/2.