
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

	orientation(mwt1gt_1,160,245). coords(mwt1gt_1,65,184,245).  visible(mwt1gt_1,245). happensAt(walking(mwt1gt_1),245). orientation(mwt1gt_2,110,245). coords(mwt1gt_2,251,22,245).  visible(mwt1gt_2,245). happensAt(walking(mwt1gt_2),245).
orientation(mwt1gt_1,160,246). coords(mwt1gt_1,66,184,246).  visible(mwt1gt_1,246). happensAt(walking(mwt1gt_1),246). orientation(mwt1gt_2,110,246). coords(mwt1gt_2,248,22,246).  visible(mwt1gt_2,246). happensAt(walking(mwt1gt_2),246).
orientation(mwt1gt_1,160,247). coords(mwt1gt_1,66,184,247).  visible(mwt1gt_1,247). happensAt(walking(mwt1gt_1),247). orientation(mwt1gt_2,110,247). coords(mwt1gt_2,247,22,247).  visible(mwt1gt_2,247). happensAt(walking(mwt1gt_2),247).
orientation(mwt1gt_1,160,248). coords(mwt1gt_1,67,183,248).  visible(mwt1gt_1,248). happensAt(walking(mwt1gt_1),248). orientation(mwt1gt_2,110,248). coords(mwt1gt_2,247,22,248).  visible(mwt1gt_2,248). happensAt(walking(mwt1gt_2),248).
orientation(mwt1gt_1,160,249). coords(mwt1gt_1,68,183,249).  visible(mwt1gt_1,249). happensAt(walking(mwt1gt_1),249). orientation(mwt1gt_2,110,249). coords(mwt1gt_2,245,24,249).  visible(mwt1gt_2,249). happensAt(walking(mwt1gt_2),249).
orientation(mwt1gt_1,160,250). coords(mwt1gt_1,69,183,250).  visible(mwt1gt_1,250). happensAt(walking(mwt1gt_1),250). orientation(mwt1gt_2,110,250). coords(mwt1gt_2,244,27,250).  visible(mwt1gt_2,250). happensAt(walking(mwt1gt_2),250).
orientation(mwt1gt_1,160,251). coords(mwt1gt_1,70,183,251).  visible(mwt1gt_1,251). happensAt(walking(mwt1gt_1),251). orientation(mwt1gt_2,110,251). coords(mwt1gt_2,243,29,251).  visible(mwt1gt_2,251). happensAt(walking(mwt1gt_2),251).
orientation(mwt1gt_1,160,252). coords(mwt1gt_1,71,183,252).  visible(mwt1gt_1,252). happensAt(walking(mwt1gt_1),252). orientation(mwt1gt_2,110,252). coords(mwt1gt_2,242,30,252).  visible(mwt1gt_2,252). happensAt(walking(mwt1gt_2),252).
orientation(mwt1gt_1,160,253). coords(mwt1gt_1,72,181,253).  visible(mwt1gt_1,253). happensAt(walking(mwt1gt_1),253). orientation(mwt1gt_2,110,253). coords(mwt1gt_2,242,31,253).  visible(mwt1gt_2,253). happensAt(walking(mwt1gt_2),253).
orientation(mwt1gt_1,160,254). coords(mwt1gt_1,72,181,254).  visible(mwt1gt_1,254). happensAt(walking(mwt1gt_1),254). orientation(mwt1gt_2,110,254). coords(mwt1gt_2,241,33,254).  visible(mwt1gt_2,254). happensAt(walking(mwt1gt_2),254).
orientation(mwt1gt_1,160,255). coords(mwt1gt_1,72,181,255).  visible(mwt1gt_1,255). happensAt(walking(mwt1gt_1),255). orientation(mwt1gt_2,110,255). coords(mwt1gt_2,241,33,255).  visible(mwt1gt_2,255). happensAt(walking(mwt1gt_2),255).
orientation(mwt1gt_1,160,256). coords(mwt1gt_1,73,180,256).  visible(mwt1gt_1,256). happensAt(walking(mwt1gt_1),256). orientation(mwt1gt_2,130,256). coords(mwt1gt_2,237,35,256).  visible(mwt1gt_2,256). happensAt(walking(mwt1gt_2),256).
orientation(mwt1gt_1,160,257). coords(mwt1gt_1,73,180,257).  visible(mwt1gt_1,257). happensAt(walking(mwt1gt_1),257). orientation(mwt1gt_2,130,257). coords(mwt1gt_2,235,35,257).  visible(mwt1gt_2,257). happensAt(walking(mwt1gt_2),257).
orientation(mwt1gt_1,160,258). coords(mwt1gt_1,73,180,258).  visible(mwt1gt_1,258). happensAt(walking(mwt1gt_1),258). orientation(mwt1gt_2,130,258). coords(mwt1gt_2,233,36,258).  visible(mwt1gt_2,258). happensAt(walking(mwt1gt_2),258).
orientation(mwt1gt_1,160,259). coords(mwt1gt_1,73,180,259).  visible(mwt1gt_1,259). happensAt(walking(mwt1gt_1),259). orientation(mwt1gt_2,130,259). coords(mwt1gt_2,233,36,259).  visible(mwt1gt_2,259). happensAt(walking(mwt1gt_2),259).
orientation(mwt1gt_1,160,260). coords(mwt1gt_1,73,179,260).  visible(mwt1gt_1,260). happensAt(walking(mwt1gt_1),260). orientation(mwt1gt_2,130,260). coords(mwt1gt_2,227,36,260).  visible(mwt1gt_2,260). happensAt(walking(mwt1gt_2),260).
orientation(mwt1gt_1,160,261). coords(mwt1gt_1,73,179,261).  visible(mwt1gt_1,261). happensAt(walking(mwt1gt_1),261). orientation(mwt1gt_2,130,261). coords(mwt1gt_2,224,38,261).  visible(mwt1gt_2,261). happensAt(walking(mwt1gt_2),261).
orientation(mwt1gt_1,160,262). coords(mwt1gt_1,75,179,262).  visible(mwt1gt_1,262). happensAt(walking(mwt1gt_1),262). orientation(mwt1gt_2,130,262). coords(mwt1gt_2,222,40,262).  visible(mwt1gt_2,262). happensAt(walking(mwt1gt_2),262).
orientation(mwt1gt_1,160,263). coords(mwt1gt_1,77,179,263).  visible(mwt1gt_1,263). happensAt(walking(mwt1gt_1),263). orientation(mwt1gt_2,130,263). coords(mwt1gt_2,220,43,263).  visible(mwt1gt_2,263). happensAt(walking(mwt1gt_2),263).
orientation(mwt1gt_1,160,264). coords(mwt1gt_1,77,179,264).  visible(mwt1gt_1,264). happensAt(walking(mwt1gt_1),264). orientation(mwt1gt_2,130,264). coords(mwt1gt_2,219,44,264).  visible(mwt1gt_2,264). happensAt(walking(mwt1gt_2),264).
orientation(mwt1gt_1,160,265). coords(mwt1gt_1,78,178,265).  visible(mwt1gt_1,265). happensAt(walking(mwt1gt_1),265). orientation(mwt1gt_2,130,265). coords(mwt1gt_2,219,47,265).  visible(mwt1gt_2,265). happensAt(walking(mwt1gt_2),265).
orientation(mwt1gt_1,160,266). coords(mwt1gt_1,78,178,266).  visible(mwt1gt_1,266). happensAt(walking(mwt1gt_1),266). orientation(mwt1gt_2,130,266). coords(mwt1gt_2,218,48,266).  visible(mwt1gt_2,266). happensAt(walking(mwt1gt_2),266).
orientation(mwt1gt_1,160,267). coords(mwt1gt_1,79,178,267).  visible(mwt1gt_1,267). happensAt(walking(mwt1gt_1),267). orientation(mwt1gt_2,130,267). coords(mwt1gt_2,218,49,267).  visible(mwt1gt_2,267). happensAt(walking(mwt1gt_2),267).
orientation(mwt1gt_1,160,268). coords(mwt1gt_1,79,176,268).  visible(mwt1gt_1,268). happensAt(walking(mwt1gt_1),268). orientation(mwt1gt_2,130,268). coords(mwt1gt_2,217,51,268).  visible(mwt1gt_2,268). happensAt(walking(mwt1gt_2),268).
orientation(mwt1gt_1,160,269). coords(mwt1gt_1,79,176,269).  visible(mwt1gt_1,269). happensAt(walking(mwt1gt_1),269). orientation(mwt1gt_2,130,269). coords(mwt1gt_2,215,51,269).  visible(mwt1gt_2,269). happensAt(walking(mwt1gt_2),269).
#show holdsAt/2.