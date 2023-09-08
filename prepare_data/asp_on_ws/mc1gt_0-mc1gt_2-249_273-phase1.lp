
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

	orientation(mc1gt_0,143,249). coords(mc1gt_0,171,156,249).  visible(mc1gt_0,249). happensAt(walking(mc1gt_0),249). orientation(mc1gt_2,146,249). coords(mc1gt_2,137,140,249).  visible(mc1gt_2,249). happensAt(walking(mc1gt_2),249).
orientation(mc1gt_0,143,250). coords(mc1gt_0,171,157,250).  visible(mc1gt_0,250). happensAt(walking(mc1gt_0),250). orientation(mc1gt_2,146,250). coords(mc1gt_2,138,142,250).  visible(mc1gt_2,250). happensAt(walking(mc1gt_2),250).
orientation(mc1gt_0,143,251). coords(mc1gt_0,171,157,251).  visible(mc1gt_0,251). happensAt(walking(mc1gt_0),251). orientation(mc1gt_2,146,251). coords(mc1gt_2,138,142,251).  visible(mc1gt_2,251). happensAt(walking(mc1gt_2),251).
orientation(mc1gt_0,144,252). coords(mc1gt_0,172,158,252).  visible(mc1gt_0,252). happensAt(walking(mc1gt_0),252). orientation(mc1gt_2,146,252). coords(mc1gt_2,139,142,252).  visible(mc1gt_2,252). happensAt(walking(mc1gt_2),252).
orientation(mc1gt_0,145,253). coords(mc1gt_0,173,159,253).  visible(mc1gt_0,253). happensAt(walking(mc1gt_0),253). orientation(mc1gt_2,146,253). coords(mc1gt_2,141,142,253).  visible(mc1gt_2,253). happensAt(walking(mc1gt_2),253).
orientation(mc1gt_0,148,254). coords(mc1gt_0,173,159,254).  visible(mc1gt_0,254). happensAt(walking(mc1gt_0),254). orientation(mc1gt_2,146,254). coords(mc1gt_2,142,144,254).  visible(mc1gt_2,254). happensAt(walking(mc1gt_2),254).
orientation(mc1gt_0,149,255). coords(mc1gt_0,175,161,255).  visible(mc1gt_0,255). happensAt(walking(mc1gt_0),255). orientation(mc1gt_2,146,255). coords(mc1gt_2,143,143,255).  visible(mc1gt_2,255). happensAt(walking(mc1gt_2),255).
orientation(mc1gt_0,149,256). coords(mc1gt_0,176,162,256).  visible(mc1gt_0,256). happensAt(walking(mc1gt_0),256). orientation(mc1gt_2,146,256). coords(mc1gt_2,144,144,256).  visible(mc1gt_2,256). happensAt(walking(mc1gt_2),256).
orientation(mc1gt_0,152,257). coords(mc1gt_0,178,165,257).  visible(mc1gt_0,257). happensAt(walking(mc1gt_0),257). orientation(mc1gt_2,146,257). coords(mc1gt_2,145,145,257).  visible(mc1gt_2,257). happensAt(walking(mc1gt_2),257).
orientation(mc1gt_0,152,258). coords(mc1gt_0,178,167,258).  visible(mc1gt_0,258). happensAt(walking(mc1gt_0),258). orientation(mc1gt_2,146,258). coords(mc1gt_2,148,145,258).  visible(mc1gt_2,258). happensAt(walking(mc1gt_2),258).
orientation(mc1gt_0,152,259). coords(mc1gt_0,179,169,259).  visible(mc1gt_0,259). happensAt(walking(mc1gt_0),259). orientation(mc1gt_2,146,259). coords(mc1gt_2,149,146,259).  visible(mc1gt_2,259). happensAt(walking(mc1gt_2),259).
orientation(mc1gt_0,152,260). coords(mc1gt_0,181,172,260).  visible(mc1gt_0,260). happensAt(walking(mc1gt_0),260). orientation(mc1gt_2,146,260). coords(mc1gt_2,149,146,260).  visible(mc1gt_2,260). happensAt(walking(mc1gt_2),260).
orientation(mc1gt_0,152,261). coords(mc1gt_0,181,173,261).  visible(mc1gt_0,261). happensAt(walking(mc1gt_0),261). orientation(mc1gt_2,146,261). coords(mc1gt_2,150,147,261).  visible(mc1gt_2,261). happensAt(walking(mc1gt_2),261).
orientation(mc1gt_0,152,262). coords(mc1gt_0,181,173,262).  visible(mc1gt_0,262). happensAt(walking(mc1gt_0),262). orientation(mc1gt_2,146,262). coords(mc1gt_2,151,148,262).  visible(mc1gt_2,262). happensAt(walking(mc1gt_2),262).
orientation(mc1gt_0,152,263). coords(mc1gt_0,182,174,263).  visible(mc1gt_0,263). happensAt(walking(mc1gt_0),263). orientation(mc1gt_2,146,263). coords(mc1gt_2,155,150,263).  visible(mc1gt_2,263). happensAt(walking(mc1gt_2),263).
orientation(mc1gt_0,152,264). coords(mc1gt_0,182,175,264).  visible(mc1gt_0,264). happensAt(walking(mc1gt_0),264). orientation(mc1gt_2,146,264). coords(mc1gt_2,156,151,264).  visible(mc1gt_2,264). happensAt(walking(mc1gt_2),264).
orientation(mc1gt_0,152,265). coords(mc1gt_0,182,176,265).  visible(mc1gt_0,265). happensAt(walking(mc1gt_0),265). orientation(mc1gt_2,148,265). coords(mc1gt_2,157,152,265).  visible(mc1gt_2,265). happensAt(walking(mc1gt_2),265).
orientation(mc1gt_0,152,266). coords(mc1gt_0,182,177,266).  visible(mc1gt_0,266). happensAt(walking(mc1gt_0),266). orientation(mc1gt_2,148,266). coords(mc1gt_2,157,153,266).  visible(mc1gt_2,266). happensAt(walking(mc1gt_2),266).
orientation(mc1gt_0,152,267). coords(mc1gt_0,182,178,267).  visible(mc1gt_0,267). happensAt(walking(mc1gt_0),267). orientation(mc1gt_2,149,267). coords(mc1gt_2,157,154,267).  visible(mc1gt_2,267). happensAt(walking(mc1gt_2),267).
orientation(mc1gt_0,152,268). coords(mc1gt_0,182,179,268).  visible(mc1gt_0,268). happensAt(walking(mc1gt_0),268). orientation(mc1gt_2,149,268). coords(mc1gt_2,158,155,268).  visible(mc1gt_2,268). happensAt(walking(mc1gt_2),268).
orientation(mc1gt_0,152,269). coords(mc1gt_0,182,182,269).  visible(mc1gt_0,269). happensAt(walking(mc1gt_0),269). orientation(mc1gt_2,152,269). coords(mc1gt_2,158,155,269).  visible(mc1gt_2,269). happensAt(walking(mc1gt_2),269).
orientation(mc1gt_0,152,270). coords(mc1gt_0,183,185,270).  visible(mc1gt_0,270). happensAt(walking(mc1gt_0),270). orientation(mc1gt_2,153,270). coords(mc1gt_2,160,156,270).  visible(mc1gt_2,270). happensAt(walking(mc1gt_2),270).
orientation(mc1gt_0,152,271). coords(mc1gt_0,183,188,271).  visible(mc1gt_0,271). happensAt(walking(mc1gt_0),271). orientation(mc1gt_2,154,271). coords(mc1gt_2,164,156,271).  visible(mc1gt_2,271). happensAt(walking(mc1gt_2),271).
orientation(mc1gt_0,156,272). coords(mc1gt_0,184,191,272).  visible(mc1gt_0,272). happensAt(walking(mc1gt_0),272). orientation(mc1gt_2,157,272). coords(mc1gt_2,166,157,272).  visible(mc1gt_2,272). happensAt(walking(mc1gt_2),272).
orientation(mc1gt_0,156,273). coords(mc1gt_0,184,192,273).  visible(mc1gt_0,273). happensAt(walking(mc1gt_0),273). orientation(mc1gt_2,153,273). coords(mc1gt_2,169,159,273).  visible(mc1gt_2,273). happensAt(walking(mc1gt_2),273).
#show holdsAt/2.