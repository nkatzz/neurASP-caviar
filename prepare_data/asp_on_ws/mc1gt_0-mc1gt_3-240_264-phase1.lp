
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

	orientation(mc1gt_0,147,240). coords(mc1gt_0,166,140,240).  visible(mc1gt_0,240). happensAt(walking(mc1gt_0),240). orientation(mc1gt_3,139,240). coords(mc1gt_3,181,94,240).  visible(mc1gt_3,240). happensAt(walking(mc1gt_3),240).
orientation(mc1gt_0,147,241). coords(mc1gt_0,167,143,241).  visible(mc1gt_0,241). happensAt(walking(mc1gt_0),241). orientation(mc1gt_3,139,241). coords(mc1gt_3,183,95,241).  visible(mc1gt_3,241). happensAt(walking(mc1gt_3),241).
orientation(mc1gt_0,147,242). coords(mc1gt_0,167,146,242).  visible(mc1gt_0,242). happensAt(walking(mc1gt_0),242). orientation(mc1gt_3,139,242). coords(mc1gt_3,185,95,242).  visible(mc1gt_3,242). happensAt(walking(mc1gt_3),242).
orientation(mc1gt_0,147,243). coords(mc1gt_0,167,149,243).  visible(mc1gt_0,243). happensAt(walking(mc1gt_0),243). orientation(mc1gt_3,139,243). coords(mc1gt_3,187,96,243).  visible(mc1gt_3,243). happensAt(walking(mc1gt_3),243).
orientation(mc1gt_0,147,244). coords(mc1gt_0,168,151,244).  visible(mc1gt_0,244). happensAt(walking(mc1gt_0),244). orientation(mc1gt_3,139,244). coords(mc1gt_3,188,97,244).  visible(mc1gt_3,244). happensAt(walking(mc1gt_3),244).
orientation(mc1gt_0,143,245). coords(mc1gt_0,169,152,245).  visible(mc1gt_0,245). happensAt(walking(mc1gt_0),245). orientation(mc1gt_3,139,245). coords(mc1gt_3,189,97,245).  visible(mc1gt_3,245). happensAt(walking(mc1gt_3),245).
orientation(mc1gt_0,143,246). coords(mc1gt_0,170,153,246).  visible(mc1gt_0,246). happensAt(walking(mc1gt_0),246). orientation(mc1gt_3,139,246). coords(mc1gt_3,190,97,246).  visible(mc1gt_3,246). happensAt(walking(mc1gt_3),246).
orientation(mc1gt_0,143,247). coords(mc1gt_0,171,155,247).  visible(mc1gt_0,247). happensAt(walking(mc1gt_0),247). orientation(mc1gt_3,139,247). coords(mc1gt_3,190,98,247).  visible(mc1gt_3,247). happensAt(walking(mc1gt_3),247).
orientation(mc1gt_0,143,248). coords(mc1gt_0,171,156,248).  visible(mc1gt_0,248). happensAt(walking(mc1gt_0),248). orientation(mc1gt_3,139,248). coords(mc1gt_3,191,98,248).  visible(mc1gt_3,248). happensAt(walking(mc1gt_3),248).
orientation(mc1gt_0,143,249). coords(mc1gt_0,171,156,249).  visible(mc1gt_0,249). happensAt(walking(mc1gt_0),249). orientation(mc1gt_3,139,249). coords(mc1gt_3,191,98,249).  visible(mc1gt_3,249). happensAt(walking(mc1gt_3),249).
orientation(mc1gt_0,143,250). coords(mc1gt_0,171,157,250).  visible(mc1gt_0,250). happensAt(walking(mc1gt_0),250). orientation(mc1gt_3,139,250). coords(mc1gt_3,192,98,250).  visible(mc1gt_3,250). happensAt(walking(mc1gt_3),250).
orientation(mc1gt_0,143,251). coords(mc1gt_0,171,157,251).  visible(mc1gt_0,251). happensAt(walking(mc1gt_0),251). orientation(mc1gt_3,139,251). coords(mc1gt_3,195,98,251).  visible(mc1gt_3,251). happensAt(walking(mc1gt_3),251).
orientation(mc1gt_0,144,252). coords(mc1gt_0,172,158,252).  visible(mc1gt_0,252). happensAt(walking(mc1gt_0),252). orientation(mc1gt_3,139,252). coords(mc1gt_3,199,99,252).  visible(mc1gt_3,252). happensAt(walking(mc1gt_3),252).
orientation(mc1gt_0,145,253). coords(mc1gt_0,173,159,253).  visible(mc1gt_0,253). happensAt(walking(mc1gt_0),253). orientation(mc1gt_3,139,253). coords(mc1gt_3,202,99,253).  visible(mc1gt_3,253). happensAt(walking(mc1gt_3),253).
orientation(mc1gt_0,148,254). coords(mc1gt_0,173,159,254).  visible(mc1gt_0,254). happensAt(walking(mc1gt_0),254). orientation(mc1gt_3,139,254). coords(mc1gt_3,204,101,254).  visible(mc1gt_3,254). happensAt(walking(mc1gt_3),254).
orientation(mc1gt_0,149,255). coords(mc1gt_0,175,161,255).  visible(mc1gt_0,255). happensAt(walking(mc1gt_0),255). orientation(mc1gt_3,139,255). coords(mc1gt_3,206,102,255).  visible(mc1gt_3,255). happensAt(walking(mc1gt_3),255).
orientation(mc1gt_0,149,256). coords(mc1gt_0,176,162,256).  visible(mc1gt_0,256). happensAt(walking(mc1gt_0),256). orientation(mc1gt_3,139,256). coords(mc1gt_3,208,104,256).  visible(mc1gt_3,256). happensAt(walking(mc1gt_3),256).
orientation(mc1gt_0,152,257). coords(mc1gt_0,178,165,257).  visible(mc1gt_0,257). happensAt(walking(mc1gt_0),257). orientation(mc1gt_3,139,257). coords(mc1gt_3,210,104,257).  visible(mc1gt_3,257). happensAt(walking(mc1gt_3),257).
orientation(mc1gt_0,152,258). coords(mc1gt_0,178,167,258).  visible(mc1gt_0,258). happensAt(walking(mc1gt_0),258). orientation(mc1gt_3,139,258). coords(mc1gt_3,211,105,258).  visible(mc1gt_3,258). happensAt(walking(mc1gt_3),258).
orientation(mc1gt_0,152,259). coords(mc1gt_0,179,169,259).  visible(mc1gt_0,259). happensAt(walking(mc1gt_0),259). orientation(mc1gt_3,139,259). coords(mc1gt_3,213,106,259).  visible(mc1gt_3,259). happensAt(walking(mc1gt_3),259).
orientation(mc1gt_0,152,260). coords(mc1gt_0,181,172,260).  visible(mc1gt_0,260). happensAt(walking(mc1gt_0),260). orientation(mc1gt_3,134,260). coords(mc1gt_3,213,106,260).  visible(mc1gt_3,260). happensAt(walking(mc1gt_3),260).
orientation(mc1gt_0,152,261). coords(mc1gt_0,181,173,261).  visible(mc1gt_0,261). happensAt(walking(mc1gt_0),261). orientation(mc1gt_3,132,261). coords(mc1gt_3,214,107,261).  visible(mc1gt_3,261). happensAt(walking(mc1gt_3),261).
orientation(mc1gt_0,152,262). coords(mc1gt_0,181,173,262).  visible(mc1gt_0,262). happensAt(walking(mc1gt_0),262). orientation(mc1gt_3,132,262). coords(mc1gt_3,215,107,262).  visible(mc1gt_3,262). happensAt(walking(mc1gt_3),262).
orientation(mc1gt_0,152,263). coords(mc1gt_0,182,174,263).  visible(mc1gt_0,263). happensAt(walking(mc1gt_0),263). orientation(mc1gt_3,132,263). coords(mc1gt_3,216,108,263).  visible(mc1gt_3,263). happensAt(walking(mc1gt_3),263).
orientation(mc1gt_0,152,264). coords(mc1gt_0,182,175,264).  visible(mc1gt_0,264). happensAt(walking(mc1gt_0),264). orientation(mc1gt_3,132,264). coords(mc1gt_3,217,111,264).  visible(mc1gt_3,264). happensAt(walking(mc1gt_3),264).
#show holdsAt/2.