
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

	orientation(mc1gt_0,147,244). coords(mc1gt_0,168,151,244).  visible(mc1gt_0,244). happensAt(walking(mc1gt_0),244). orientation(mc1gt_1,145,244). coords(mc1gt_1,197,145,244).  visible(mc1gt_1,244). happensAt(walking(mc1gt_1),244).
orientation(mc1gt_0,143,245). coords(mc1gt_0,169,152,245).  visible(mc1gt_0,245). happensAt(walking(mc1gt_0),245). orientation(mc1gt_1,142,245). coords(mc1gt_1,198,149,245).  visible(mc1gt_1,245). happensAt(walking(mc1gt_1),245).
orientation(mc1gt_0,143,246). coords(mc1gt_0,170,153,246).  visible(mc1gt_0,246). happensAt(walking(mc1gt_0),246). orientation(mc1gt_1,142,246). coords(mc1gt_1,200,151,246).  visible(mc1gt_1,246). happensAt(walking(mc1gt_1),246).
orientation(mc1gt_0,143,247). coords(mc1gt_0,171,155,247).  visible(mc1gt_0,247). happensAt(walking(mc1gt_0),247). orientation(mc1gt_1,142,247). coords(mc1gt_1,201,154,247).  visible(mc1gt_1,247). happensAt(walking(mc1gt_1),247).
orientation(mc1gt_0,143,248). coords(mc1gt_0,171,156,248).  visible(mc1gt_0,248). happensAt(walking(mc1gt_0),248). orientation(mc1gt_1,142,248). coords(mc1gt_1,202,155,248).  visible(mc1gt_1,248). happensAt(walking(mc1gt_1),248).
orientation(mc1gt_0,143,249). coords(mc1gt_0,171,156,249).  visible(mc1gt_0,249). happensAt(walking(mc1gt_0),249). orientation(mc1gt_1,142,249). coords(mc1gt_1,203,157,249).  visible(mc1gt_1,249). happensAt(walking(mc1gt_1),249).
orientation(mc1gt_0,143,250). coords(mc1gt_0,171,157,250).  visible(mc1gt_0,250). happensAt(walking(mc1gt_0),250). orientation(mc1gt_1,142,250). coords(mc1gt_1,204,158,250).  visible(mc1gt_1,250). happensAt(walking(mc1gt_1),250).
orientation(mc1gt_0,143,251). coords(mc1gt_0,171,157,251).  visible(mc1gt_0,251). happensAt(walking(mc1gt_0),251). orientation(mc1gt_1,142,251). coords(mc1gt_1,205,159,251).  visible(mc1gt_1,251). happensAt(walking(mc1gt_1),251).
orientation(mc1gt_0,144,252). coords(mc1gt_0,172,158,252).  visible(mc1gt_0,252). happensAt(walking(mc1gt_0),252). orientation(mc1gt_1,144,252). coords(mc1gt_1,205,159,252).  visible(mc1gt_1,252). happensAt(walking(mc1gt_1),252).
orientation(mc1gt_0,145,253). coords(mc1gt_0,173,159,253).  visible(mc1gt_0,253). happensAt(walking(mc1gt_0),253). orientation(mc1gt_1,145,253). coords(mc1gt_1,206,160,253).  visible(mc1gt_1,253). happensAt(walking(mc1gt_1),253).
orientation(mc1gt_0,148,254). coords(mc1gt_0,173,159,254).  visible(mc1gt_0,254). happensAt(walking(mc1gt_0),254). orientation(mc1gt_1,146,254). coords(mc1gt_1,206,161,254).  visible(mc1gt_1,254). happensAt(walking(mc1gt_1),254).
orientation(mc1gt_0,149,255). coords(mc1gt_0,175,161,255).  visible(mc1gt_0,255). happensAt(walking(mc1gt_0),255). orientation(mc1gt_1,147,255). coords(mc1gt_1,206,162,255).  visible(mc1gt_1,255). happensAt(walking(mc1gt_1),255).
orientation(mc1gt_0,149,256). coords(mc1gt_0,176,162,256).  visible(mc1gt_0,256). happensAt(walking(mc1gt_0),256). orientation(mc1gt_1,146,256). coords(mc1gt_1,206,163,256).  visible(mc1gt_1,256). happensAt(walking(mc1gt_1),256).
orientation(mc1gt_0,152,257). coords(mc1gt_0,178,165,257).  visible(mc1gt_0,257). happensAt(walking(mc1gt_0),257). orientation(mc1gt_1,147,257). coords(mc1gt_1,206,163,257).  visible(mc1gt_1,257). happensAt(walking(mc1gt_1),257).
orientation(mc1gt_0,152,258). coords(mc1gt_0,178,167,258).  visible(mc1gt_0,258). happensAt(walking(mc1gt_0),258). orientation(mc1gt_1,153,258). coords(mc1gt_1,207,166,258).  visible(mc1gt_1,258). happensAt(walking(mc1gt_1),258).
orientation(mc1gt_0,152,259). coords(mc1gt_0,179,169,259).  visible(mc1gt_0,259). happensAt(walking(mc1gt_0),259). orientation(mc1gt_1,155,259). coords(mc1gt_1,207,170,259).  visible(mc1gt_1,259). happensAt(walking(mc1gt_1),259).
orientation(mc1gt_0,152,260). coords(mc1gt_0,181,172,260).  visible(mc1gt_0,260). happensAt(walking(mc1gt_0),260). orientation(mc1gt_1,159,260). coords(mc1gt_1,207,174,260).  visible(mc1gt_1,260). happensAt(walking(mc1gt_1),260).
orientation(mc1gt_0,152,261). coords(mc1gt_0,181,173,261).  visible(mc1gt_0,261). happensAt(walking(mc1gt_0),261). orientation(mc1gt_1,164,261). coords(mc1gt_1,207,178,261).  visible(mc1gt_1,261). happensAt(walking(mc1gt_1),261).
orientation(mc1gt_0,152,262). coords(mc1gt_0,181,173,262).  visible(mc1gt_0,262). happensAt(walking(mc1gt_0),262). orientation(mc1gt_1,165,262). coords(mc1gt_1,208,181,262).  visible(mc1gt_1,262). happensAt(walking(mc1gt_1),262).
orientation(mc1gt_0,152,263). coords(mc1gt_0,182,174,263).  visible(mc1gt_0,263). happensAt(walking(mc1gt_0),263). orientation(mc1gt_1,168,263). coords(mc1gt_1,208,183,263).  visible(mc1gt_1,263). happensAt(walking(mc1gt_1),263).
orientation(mc1gt_0,152,264). coords(mc1gt_0,182,175,264).  visible(mc1gt_0,264). happensAt(walking(mc1gt_0),264). orientation(mc1gt_1,171,264). coords(mc1gt_1,209,185,264).  visible(mc1gt_1,264). happensAt(walking(mc1gt_1),264).
orientation(mc1gt_0,152,265). coords(mc1gt_0,182,176,265).  visible(mc1gt_0,265). happensAt(walking(mc1gt_0),265). orientation(mc1gt_1,172,265). coords(mc1gt_1,209,186,265).  visible(mc1gt_1,265). happensAt(walking(mc1gt_1),265).
orientation(mc1gt_0,152,266). coords(mc1gt_0,182,177,266).  visible(mc1gt_0,266). happensAt(walking(mc1gt_0),266). orientation(mc1gt_1,174,266). coords(mc1gt_1,209,186,266).  visible(mc1gt_1,266). happensAt(walking(mc1gt_1),266).
orientation(mc1gt_0,152,267). coords(mc1gt_0,182,178,267).  visible(mc1gt_0,267). happensAt(walking(mc1gt_0),267). orientation(mc1gt_1,174,267). coords(mc1gt_1,209,186,267).  visible(mc1gt_1,267). happensAt(walking(mc1gt_1),267).
orientation(mc1gt_0,152,268). coords(mc1gt_0,182,179,268).  visible(mc1gt_0,268). happensAt(walking(mc1gt_0),268). orientation(mc1gt_1,175,268). coords(mc1gt_1,209,187,268).  visible(mc1gt_1,268). happensAt(walking(mc1gt_1),268).
#show holdsAt/2.