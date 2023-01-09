
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

	orientation(mws1gt_5,161,240). coords(mws1gt_5,43,134,240).  visible(mws1gt_5,240). happensAt(walking(mws1gt_5),240). orientation(mws1gt_6,146,240). coords(mws1gt_6,44,110,240).  visible(mws1gt_6,240). happensAt(walking(mws1gt_6),240).
orientation(mws1gt_5,161,241). coords(mws1gt_5,43,134,241).  visible(mws1gt_5,241). happensAt(walking(mws1gt_5),241). orientation(mws1gt_6,146,241). coords(mws1gt_6,44,111,241).  visible(mws1gt_6,241). happensAt(walking(mws1gt_6),241).
orientation(mws1gt_5,161,242). coords(mws1gt_5,42,135,242).  visible(mws1gt_5,242). happensAt(walking(mws1gt_5),242). orientation(mws1gt_6,146,242). coords(mws1gt_6,43,112,242).  visible(mws1gt_6,242). happensAt(walking(mws1gt_6),242).
orientation(mws1gt_5,161,243). coords(mws1gt_5,42,135,243).  visible(mws1gt_5,243). happensAt(walking(mws1gt_5),243). orientation(mws1gt_6,146,243). coords(mws1gt_6,43,112,243).  visible(mws1gt_6,243). happensAt(walking(mws1gt_6),243).
orientation(mws1gt_5,161,244). coords(mws1gt_5,42,135,244).  visible(mws1gt_5,244). happensAt(walking(mws1gt_5),244). orientation(mws1gt_6,146,244). coords(mws1gt_6,43,112,244).  visible(mws1gt_6,244). happensAt(walking(mws1gt_6),244).
orientation(mws1gt_5,161,245). coords(mws1gt_5,42,135,245).  visible(mws1gt_5,245). happensAt(walking(mws1gt_5),245). orientation(mws1gt_6,146,245). coords(mws1gt_6,43,112,245).  visible(mws1gt_6,245). happensAt(walking(mws1gt_6),245).
orientation(mws1gt_5,161,246). coords(mws1gt_5,41,135,246).  visible(mws1gt_5,246). happensAt(walking(mws1gt_5),246). orientation(mws1gt_6,146,246). coords(mws1gt_6,43,112,246).  visible(mws1gt_6,246). happensAt(walking(mws1gt_6),246).
orientation(mws1gt_5,161,247). coords(mws1gt_5,41,137,247).  visible(mws1gt_5,247). happensAt(walking(mws1gt_5),247). orientation(mws1gt_6,146,247). coords(mws1gt_6,43,112,247).  visible(mws1gt_6,247). happensAt(walking(mws1gt_6),247).
orientation(mws1gt_5,161,248). coords(mws1gt_5,41,138,248).  visible(mws1gt_5,248). happensAt(walking(mws1gt_5),248). orientation(mws1gt_6,146,248). coords(mws1gt_6,43,112,248).  visible(mws1gt_6,248). happensAt(walking(mws1gt_6),248).
orientation(mws1gt_5,161,249). coords(mws1gt_5,41,138,249).  visible(mws1gt_5,249). happensAt(walking(mws1gt_5),249). orientation(mws1gt_6,146,249). coords(mws1gt_6,43,114,249).  visible(mws1gt_6,249). happensAt(walking(mws1gt_6),249).
orientation(mws1gt_5,161,250). coords(mws1gt_5,41,139,250).  visible(mws1gt_5,250). happensAt(walking(mws1gt_5),250). orientation(mws1gt_6,146,250). coords(mws1gt_6,43,115,250).  visible(mws1gt_6,250). happensAt(walking(mws1gt_6),250).
orientation(mws1gt_5,161,251). coords(mws1gt_5,41,140,251).  visible(mws1gt_5,251). happensAt(walking(mws1gt_5),251). orientation(mws1gt_6,146,251). coords(mws1gt_6,42,116,251).  visible(mws1gt_6,251). happensAt(walking(mws1gt_6),251).
orientation(mws1gt_5,161,252). coords(mws1gt_5,41,141,252).  visible(mws1gt_5,252). happensAt(walking(mws1gt_5),252). orientation(mws1gt_6,146,252). coords(mws1gt_6,42,117,252).  visible(mws1gt_6,252). happensAt(walking(mws1gt_6),252).
orientation(mws1gt_5,161,253). coords(mws1gt_5,41,142,253).  visible(mws1gt_5,253). happensAt(walking(mws1gt_5),253). orientation(mws1gt_6,146,253). coords(mws1gt_6,42,119,253).  visible(mws1gt_6,253). happensAt(walking(mws1gt_6),253).
orientation(mws1gt_5,161,254). coords(mws1gt_5,41,143,254).  visible(mws1gt_5,254). happensAt(walking(mws1gt_5),254). orientation(mws1gt_6,146,254). coords(mws1gt_6,42,119,254).  visible(mws1gt_6,254). happensAt(walking(mws1gt_6),254).
orientation(mws1gt_5,161,255). coords(mws1gt_5,41,143,255).  visible(mws1gt_5,255). happensAt(walking(mws1gt_5),255). orientation(mws1gt_6,146,255). coords(mws1gt_6,43,120,255).  visible(mws1gt_6,255). happensAt(walking(mws1gt_6),255).
orientation(mws1gt_5,161,256). coords(mws1gt_5,41,143,256).  visible(mws1gt_5,256). happensAt(walking(mws1gt_5),256). orientation(mws1gt_6,146,256). coords(mws1gt_6,43,120,256).  visible(mws1gt_6,256). happensAt(walking(mws1gt_6),256).
orientation(mws1gt_5,161,257). coords(mws1gt_5,41,143,257).  visible(mws1gt_5,257). happensAt(walking(mws1gt_5),257). orientation(mws1gt_6,146,257). coords(mws1gt_6,43,121,257).  visible(mws1gt_6,257). happensAt(walking(mws1gt_6),257).
orientation(mws1gt_5,161,258). coords(mws1gt_5,41,143,258).  visible(mws1gt_5,258). happensAt(walking(mws1gt_5),258). orientation(mws1gt_6,146,258). coords(mws1gt_6,43,121,258).  visible(mws1gt_6,258). happensAt(walking(mws1gt_6),258).
orientation(mws1gt_5,161,259). coords(mws1gt_5,41,144,259).  visible(mws1gt_5,259). happensAt(walking(mws1gt_5),259). orientation(mws1gt_6,146,259). coords(mws1gt_6,43,124,259).  visible(mws1gt_6,259). happensAt(walking(mws1gt_6),259).
orientation(mws1gt_5,161,260). coords(mws1gt_5,41,147,260).  visible(mws1gt_5,260). happensAt(walking(mws1gt_5),260). orientation(mws1gt_6,146,260). coords(mws1gt_6,43,125,260).  visible(mws1gt_6,260). happensAt(walking(mws1gt_6),260).
orientation(mws1gt_5,161,261). coords(mws1gt_5,41,148,261).  visible(mws1gt_5,261). happensAt(walking(mws1gt_5),261). orientation(mws1gt_6,146,261). coords(mws1gt_6,43,126,261).  visible(mws1gt_6,261). happensAt(walking(mws1gt_6),261).
orientation(mws1gt_5,161,262). coords(mws1gt_5,41,150,262).  visible(mws1gt_5,262). happensAt(walking(mws1gt_5),262). orientation(mws1gt_6,146,262). coords(mws1gt_6,43,128,262).  visible(mws1gt_6,262). happensAt(walking(mws1gt_6),262).
orientation(mws1gt_5,161,263). coords(mws1gt_5,41,150,263).  visible(mws1gt_5,263). happensAt(walking(mws1gt_5),263). orientation(mws1gt_6,146,263). coords(mws1gt_6,42,129,263).  visible(mws1gt_6,263). happensAt(walking(mws1gt_6),263).
orientation(mws1gt_5,161,264). coords(mws1gt_5,41,151,264).  visible(mws1gt_5,264). happensAt(walking(mws1gt_5),264). orientation(mws1gt_6,146,264). coords(mws1gt_6,42,129,264).  visible(mws1gt_6,264). happensAt(walking(mws1gt_6),264).
#show holdsAt/2.