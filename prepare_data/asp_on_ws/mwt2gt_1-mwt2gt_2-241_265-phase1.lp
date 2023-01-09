
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

	orientation(mwt2gt_1,138,241). coords(mwt2gt_1,140,68,241).  visible(mwt2gt_1,241). happensAt(walking(mwt2gt_1),241). orientation(mwt2gt_2,146,241). coords(mwt2gt_2,210,163,241).  visible(mwt2gt_2,241). happensAt(walking(mwt2gt_2),241).
orientation(mwt2gt_1,138,242). coords(mwt2gt_1,140,68,242).  visible(mwt2gt_1,242). happensAt(walking(mwt2gt_1),242). orientation(mwt2gt_2,146,242). coords(mwt2gt_2,207,160,242).  visible(mwt2gt_2,242). happensAt(walking(mwt2gt_2),242).
orientation(mwt2gt_1,138,243). coords(mwt2gt_1,140,68,243).  visible(mwt2gt_1,243). happensAt(walking(mwt2gt_1),243). orientation(mwt2gt_2,146,243). coords(mwt2gt_2,206,157,243).  visible(mwt2gt_2,243). happensAt(walking(mwt2gt_2),243).
orientation(mwt2gt_1,138,244). coords(mwt2gt_1,140,68,244).  visible(mwt2gt_1,244). happensAt(walking(mwt2gt_1),244). orientation(mwt2gt_2,148,244). coords(mwt2gt_2,204,154,244).  visible(mwt2gt_2,244). happensAt(walking(mwt2gt_2),244).
orientation(mwt2gt_1,138,245). coords(mwt2gt_1,140,69,245).  visible(mwt2gt_1,245). happensAt(walking(mwt2gt_1),245). orientation(mwt2gt_2,148,245). coords(mwt2gt_2,202,154,245).  visible(mwt2gt_2,245). happensAt(walking(mwt2gt_2),245).
orientation(mwt2gt_1,138,246). coords(mwt2gt_1,140,68,246).  visible(mwt2gt_1,246). happensAt(walking(mwt2gt_1),246). orientation(mwt2gt_2,148,246). coords(mwt2gt_2,201,153,246).  visible(mwt2gt_2,246). happensAt(walking(mwt2gt_2),246).
orientation(mwt2gt_1,138,247). coords(mwt2gt_1,140,68,247).  visible(mwt2gt_1,247). happensAt(walking(mwt2gt_1),247). orientation(mwt2gt_2,148,247). coords(mwt2gt_2,201,153,247).  visible(mwt2gt_2,247). happensAt(walking(mwt2gt_2),247).
orientation(mwt2gt_1,138,248). coords(mwt2gt_1,139,69,248).  visible(mwt2gt_1,248). happensAt(walking(mwt2gt_1),248). orientation(mwt2gt_2,148,248). coords(mwt2gt_2,200,151,248).  visible(mwt2gt_2,248). happensAt(walking(mwt2gt_2),248).
orientation(mwt2gt_1,138,249). coords(mwt2gt_1,139,69,249).  visible(mwt2gt_1,249). happensAt(walking(mwt2gt_1),249). orientation(mwt2gt_2,148,249). coords(mwt2gt_2,200,150,249).  visible(mwt2gt_2,249). happensAt(walking(mwt2gt_2),249).
orientation(mwt2gt_1,138,250). coords(mwt2gt_1,139,69,250).  visible(mwt2gt_1,250). happensAt(walking(mwt2gt_1),250). orientation(mwt2gt_2,145,250). coords(mwt2gt_2,199,150,250).  visible(mwt2gt_2,250). happensAt(walking(mwt2gt_2),250).
orientation(mwt2gt_1,138,251). coords(mwt2gt_1,139,69,251).  visible(mwt2gt_1,251). happensAt(walking(mwt2gt_1),251). orientation(mwt2gt_2,145,251). coords(mwt2gt_2,199,148,251).  visible(mwt2gt_2,251). happensAt(walking(mwt2gt_2),251).
orientation(mwt2gt_1,138,252). coords(mwt2gt_1,140,70,252).  visible(mwt2gt_1,252). happensAt(walking(mwt2gt_1),252). orientation(mwt2gt_2,145,252). coords(mwt2gt_2,198,147,252).  visible(mwt2gt_2,252). happensAt(walking(mwt2gt_2),252).
orientation(mwt2gt_1,138,253). coords(mwt2gt_1,139,71,253).  visible(mwt2gt_1,253). happensAt(walking(mwt2gt_1),253). orientation(mwt2gt_2,145,253). coords(mwt2gt_2,196,145,253).  visible(mwt2gt_2,253). happensAt(walking(mwt2gt_2),253).
orientation(mwt2gt_1,138,254). coords(mwt2gt_1,139,72,254).  visible(mwt2gt_1,254). happensAt(walking(mwt2gt_1),254). orientation(mwt2gt_2,145,254). coords(mwt2gt_2,194,143,254).  visible(mwt2gt_2,254). happensAt(walking(mwt2gt_2),254).
orientation(mwt2gt_1,138,255). coords(mwt2gt_1,139,73,255).  visible(mwt2gt_1,255). happensAt(walking(mwt2gt_1),255). orientation(mwt2gt_2,141,255). coords(mwt2gt_2,192,140,255).  visible(mwt2gt_2,255). happensAt(walking(mwt2gt_2),255).
orientation(mwt2gt_1,138,256). coords(mwt2gt_1,139,73,256).  visible(mwt2gt_1,256). happensAt(walking(mwt2gt_1),256). orientation(mwt2gt_2,141,256). coords(mwt2gt_2,190,137,256).  visible(mwt2gt_2,256). happensAt(walking(mwt2gt_2),256).
orientation(mwt2gt_1,138,257). coords(mwt2gt_1,139,74,257).  visible(mwt2gt_1,257). happensAt(walking(mwt2gt_1),257). orientation(mwt2gt_2,141,257). coords(mwt2gt_2,190,135,257).  visible(mwt2gt_2,257). happensAt(walking(mwt2gt_2),257).
orientation(mwt2gt_1,138,258). coords(mwt2gt_1,139,74,258).  visible(mwt2gt_1,258). happensAt(walking(mwt2gt_1),258). orientation(mwt2gt_2,141,258). coords(mwt2gt_2,189,133,258).  visible(mwt2gt_2,258). happensAt(walking(mwt2gt_2),258).
orientation(mwt2gt_1,143,259). coords(mwt2gt_1,139,74,259).  visible(mwt2gt_1,259). happensAt(walking(mwt2gt_1),259). orientation(mwt2gt_2,141,259). coords(mwt2gt_2,189,132,259).  visible(mwt2gt_2,259). happensAt(walking(mwt2gt_2),259).
orientation(mwt2gt_1,143,260). coords(mwt2gt_1,139,74,260).  visible(mwt2gt_1,260). happensAt(walking(mwt2gt_1),260). orientation(mwt2gt_2,141,260). coords(mwt2gt_2,188,132,260).  visible(mwt2gt_2,260). happensAt(walking(mwt2gt_2),260).
orientation(mwt2gt_1,143,261). coords(mwt2gt_1,139,74,261).  visible(mwt2gt_1,261). happensAt(walking(mwt2gt_1),261). orientation(mwt2gt_2,145,261). coords(mwt2gt_2,188,131,261).  visible(mwt2gt_2,261). happensAt(walking(mwt2gt_2),261).
orientation(mwt2gt_1,143,262). coords(mwt2gt_1,139,74,262).  visible(mwt2gt_1,262). happensAt(walking(mwt2gt_1),262). orientation(mwt2gt_2,145,262). coords(mwt2gt_2,188,131,262).  visible(mwt2gt_2,262). happensAt(walking(mwt2gt_2),262).
orientation(mwt2gt_1,143,263). coords(mwt2gt_1,138,74,263).  visible(mwt2gt_1,263). happensAt(walking(mwt2gt_1),263). orientation(mwt2gt_2,145,263). coords(mwt2gt_2,187,130,263).  visible(mwt2gt_2,263). happensAt(walking(mwt2gt_2),263).
orientation(mwt2gt_1,143,264). coords(mwt2gt_1,138,75,264).  visible(mwt2gt_1,264). happensAt(walking(mwt2gt_1),264). orientation(mwt2gt_2,145,264). coords(mwt2gt_2,185,129,264).  visible(mwt2gt_2,264). happensAt(walking(mwt2gt_2),264).
orientation(mwt2gt_1,143,265). coords(mwt2gt_1,138,75,265).  visible(mwt2gt_1,265). happensAt(walking(mwt2gt_1),265). orientation(mwt2gt_2,145,265). coords(mwt2gt_2,184,128,265).  visible(mwt2gt_2,265). happensAt(walking(mwt2gt_2),265).
#show holdsAt/2.