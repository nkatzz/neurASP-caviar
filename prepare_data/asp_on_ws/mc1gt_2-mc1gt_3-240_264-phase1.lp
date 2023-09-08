
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

	orientation(mc1gt_2,146,240). coords(mc1gt_2,133,137,240).  visible(mc1gt_2,240). happensAt(walking(mc1gt_2),240). orientation(mc1gt_3,139,240). coords(mc1gt_3,181,94,240).  visible(mc1gt_3,240). happensAt(walking(mc1gt_3),240).
orientation(mc1gt_2,146,241). coords(mc1gt_2,133,137,241).  visible(mc1gt_2,241). happensAt(walking(mc1gt_2),241). orientation(mc1gt_3,139,241). coords(mc1gt_3,183,95,241).  visible(mc1gt_3,241). happensAt(walking(mc1gt_3),241).
orientation(mc1gt_2,146,242). coords(mc1gt_2,133,137,242).  visible(mc1gt_2,242). happensAt(walking(mc1gt_2),242). orientation(mc1gt_3,139,242). coords(mc1gt_3,185,95,242).  visible(mc1gt_3,242). happensAt(walking(mc1gt_3),242).
orientation(mc1gt_2,146,243). coords(mc1gt_2,134,137,243).  visible(mc1gt_2,243). happensAt(walking(mc1gt_2),243). orientation(mc1gt_3,139,243). coords(mc1gt_3,187,96,243).  visible(mc1gt_3,243). happensAt(walking(mc1gt_3),243).
orientation(mc1gt_2,146,244). coords(mc1gt_2,134,138,244).  visible(mc1gt_2,244). happensAt(walking(mc1gt_2),244). orientation(mc1gt_3,139,244). coords(mc1gt_3,188,97,244).  visible(mc1gt_3,244). happensAt(walking(mc1gt_3),244).
orientation(mc1gt_2,146,245). coords(mc1gt_2,135,138,245).  visible(mc1gt_2,245). happensAt(walking(mc1gt_2),245). orientation(mc1gt_3,139,245). coords(mc1gt_3,189,97,245).  visible(mc1gt_3,245). happensAt(walking(mc1gt_3),245).
orientation(mc1gt_2,146,246). coords(mc1gt_2,135,139,246).  visible(mc1gt_2,246). happensAt(walking(mc1gt_2),246). orientation(mc1gt_3,139,246). coords(mc1gt_3,190,97,246).  visible(mc1gt_3,246). happensAt(walking(mc1gt_3),246).
orientation(mc1gt_2,146,247). coords(mc1gt_2,136,139,247).  visible(mc1gt_2,247). happensAt(walking(mc1gt_2),247). orientation(mc1gt_3,139,247). coords(mc1gt_3,190,98,247).  visible(mc1gt_3,247). happensAt(walking(mc1gt_3),247).
orientation(mc1gt_2,146,248). coords(mc1gt_2,137,140,248).  visible(mc1gt_2,248). happensAt(walking(mc1gt_2),248). orientation(mc1gt_3,139,248). coords(mc1gt_3,191,98,248).  visible(mc1gt_3,248). happensAt(walking(mc1gt_3),248).
orientation(mc1gt_2,146,249). coords(mc1gt_2,137,140,249).  visible(mc1gt_2,249). happensAt(walking(mc1gt_2),249). orientation(mc1gt_3,139,249). coords(mc1gt_3,191,98,249).  visible(mc1gt_3,249). happensAt(walking(mc1gt_3),249).
orientation(mc1gt_2,146,250). coords(mc1gt_2,138,142,250).  visible(mc1gt_2,250). happensAt(walking(mc1gt_2),250). orientation(mc1gt_3,139,250). coords(mc1gt_3,192,98,250).  visible(mc1gt_3,250). happensAt(walking(mc1gt_3),250).
orientation(mc1gt_2,146,251). coords(mc1gt_2,138,142,251).  visible(mc1gt_2,251). happensAt(walking(mc1gt_2),251). orientation(mc1gt_3,139,251). coords(mc1gt_3,195,98,251).  visible(mc1gt_3,251). happensAt(walking(mc1gt_3),251).
orientation(mc1gt_2,146,252). coords(mc1gt_2,139,142,252).  visible(mc1gt_2,252). happensAt(walking(mc1gt_2),252). orientation(mc1gt_3,139,252). coords(mc1gt_3,199,99,252).  visible(mc1gt_3,252). happensAt(walking(mc1gt_3),252).
orientation(mc1gt_2,146,253). coords(mc1gt_2,141,142,253).  visible(mc1gt_2,253). happensAt(walking(mc1gt_2),253). orientation(mc1gt_3,139,253). coords(mc1gt_3,202,99,253).  visible(mc1gt_3,253). happensAt(walking(mc1gt_3),253).
orientation(mc1gt_2,146,254). coords(mc1gt_2,142,144,254).  visible(mc1gt_2,254). happensAt(walking(mc1gt_2),254). orientation(mc1gt_3,139,254). coords(mc1gt_3,204,101,254).  visible(mc1gt_3,254). happensAt(walking(mc1gt_3),254).
orientation(mc1gt_2,146,255). coords(mc1gt_2,143,143,255).  visible(mc1gt_2,255). happensAt(walking(mc1gt_2),255). orientation(mc1gt_3,139,255). coords(mc1gt_3,206,102,255).  visible(mc1gt_3,255). happensAt(walking(mc1gt_3),255).
orientation(mc1gt_2,146,256). coords(mc1gt_2,144,144,256).  visible(mc1gt_2,256). happensAt(walking(mc1gt_2),256). orientation(mc1gt_3,139,256). coords(mc1gt_3,208,104,256).  visible(mc1gt_3,256). happensAt(walking(mc1gt_3),256).
orientation(mc1gt_2,146,257). coords(mc1gt_2,145,145,257).  visible(mc1gt_2,257). happensAt(walking(mc1gt_2),257). orientation(mc1gt_3,139,257). coords(mc1gt_3,210,104,257).  visible(mc1gt_3,257). happensAt(walking(mc1gt_3),257).
orientation(mc1gt_2,146,258). coords(mc1gt_2,148,145,258).  visible(mc1gt_2,258). happensAt(walking(mc1gt_2),258). orientation(mc1gt_3,139,258). coords(mc1gt_3,211,105,258).  visible(mc1gt_3,258). happensAt(walking(mc1gt_3),258).
orientation(mc1gt_2,146,259). coords(mc1gt_2,149,146,259).  visible(mc1gt_2,259). happensAt(walking(mc1gt_2),259). orientation(mc1gt_3,139,259). coords(mc1gt_3,213,106,259).  visible(mc1gt_3,259). happensAt(walking(mc1gt_3),259).
orientation(mc1gt_2,146,260). coords(mc1gt_2,149,146,260).  visible(mc1gt_2,260). happensAt(walking(mc1gt_2),260). orientation(mc1gt_3,134,260). coords(mc1gt_3,213,106,260).  visible(mc1gt_3,260). happensAt(walking(mc1gt_3),260).
orientation(mc1gt_2,146,261). coords(mc1gt_2,150,147,261).  visible(mc1gt_2,261). happensAt(walking(mc1gt_2),261). orientation(mc1gt_3,132,261). coords(mc1gt_3,214,107,261).  visible(mc1gt_3,261). happensAt(walking(mc1gt_3),261).
orientation(mc1gt_2,146,262). coords(mc1gt_2,151,148,262).  visible(mc1gt_2,262). happensAt(walking(mc1gt_2),262). orientation(mc1gt_3,132,262). coords(mc1gt_3,215,107,262).  visible(mc1gt_3,262). happensAt(walking(mc1gt_3),262).
orientation(mc1gt_2,146,263). coords(mc1gt_2,155,150,263).  visible(mc1gt_2,263). happensAt(walking(mc1gt_2),263). orientation(mc1gt_3,132,263). coords(mc1gt_3,216,108,263).  visible(mc1gt_3,263). happensAt(walking(mc1gt_3),263).
orientation(mc1gt_2,146,264). coords(mc1gt_2,156,151,264).  visible(mc1gt_2,264). happensAt(walking(mc1gt_2),264). orientation(mc1gt_3,132,264). coords(mc1gt_3,217,111,264).  visible(mc1gt_3,264). happensAt(walking(mc1gt_3),264).
#show holdsAt/2.