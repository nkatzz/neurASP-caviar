
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

	orientation(mc1gt_1,142,249). coords(mc1gt_1,203,157,249).  visible(mc1gt_1,249). happensAt(walking(mc1gt_1),249). orientation(mc1gt_2,146,249). coords(mc1gt_2,137,140,249).  visible(mc1gt_2,249). happensAt(walking(mc1gt_2),249).
orientation(mc1gt_1,142,250). coords(mc1gt_1,204,158,250).  visible(mc1gt_1,250). happensAt(walking(mc1gt_1),250). orientation(mc1gt_2,146,250). coords(mc1gt_2,138,142,250).  visible(mc1gt_2,250). happensAt(walking(mc1gt_2),250).
orientation(mc1gt_1,142,251). coords(mc1gt_1,205,159,251).  visible(mc1gt_1,251). happensAt(walking(mc1gt_1),251). orientation(mc1gt_2,146,251). coords(mc1gt_2,138,142,251).  visible(mc1gt_2,251). happensAt(walking(mc1gt_2),251).
orientation(mc1gt_1,144,252). coords(mc1gt_1,205,159,252).  visible(mc1gt_1,252). happensAt(walking(mc1gt_1),252). orientation(mc1gt_2,146,252). coords(mc1gt_2,139,142,252).  visible(mc1gt_2,252). happensAt(walking(mc1gt_2),252).
orientation(mc1gt_1,145,253). coords(mc1gt_1,206,160,253).  visible(mc1gt_1,253). happensAt(walking(mc1gt_1),253). orientation(mc1gt_2,146,253). coords(mc1gt_2,141,142,253).  visible(mc1gt_2,253). happensAt(walking(mc1gt_2),253).
orientation(mc1gt_1,146,254). coords(mc1gt_1,206,161,254).  visible(mc1gt_1,254). happensAt(walking(mc1gt_1),254). orientation(mc1gt_2,146,254). coords(mc1gt_2,142,144,254).  visible(mc1gt_2,254). happensAt(walking(mc1gt_2),254).
orientation(mc1gt_1,147,255). coords(mc1gt_1,206,162,255).  visible(mc1gt_1,255). happensAt(walking(mc1gt_1),255). orientation(mc1gt_2,146,255). coords(mc1gt_2,143,143,255).  visible(mc1gt_2,255). happensAt(walking(mc1gt_2),255).
orientation(mc1gt_1,146,256). coords(mc1gt_1,206,163,256).  visible(mc1gt_1,256). happensAt(walking(mc1gt_1),256). orientation(mc1gt_2,146,256). coords(mc1gt_2,144,144,256).  visible(mc1gt_2,256). happensAt(walking(mc1gt_2),256).
orientation(mc1gt_1,147,257). coords(mc1gt_1,206,163,257).  visible(mc1gt_1,257). happensAt(walking(mc1gt_1),257). orientation(mc1gt_2,146,257). coords(mc1gt_2,145,145,257).  visible(mc1gt_2,257). happensAt(walking(mc1gt_2),257).
orientation(mc1gt_1,153,258). coords(mc1gt_1,207,166,258).  visible(mc1gt_1,258). happensAt(walking(mc1gt_1),258). orientation(mc1gt_2,146,258). coords(mc1gt_2,148,145,258).  visible(mc1gt_2,258). happensAt(walking(mc1gt_2),258).
orientation(mc1gt_1,155,259). coords(mc1gt_1,207,170,259).  visible(mc1gt_1,259). happensAt(walking(mc1gt_1),259). orientation(mc1gt_2,146,259). coords(mc1gt_2,149,146,259).  visible(mc1gt_2,259). happensAt(walking(mc1gt_2),259).
orientation(mc1gt_1,159,260). coords(mc1gt_1,207,174,260).  visible(mc1gt_1,260). happensAt(walking(mc1gt_1),260). orientation(mc1gt_2,146,260). coords(mc1gt_2,149,146,260).  visible(mc1gt_2,260). happensAt(walking(mc1gt_2),260).
orientation(mc1gt_1,164,261). coords(mc1gt_1,207,178,261).  visible(mc1gt_1,261). happensAt(walking(mc1gt_1),261). orientation(mc1gt_2,146,261). coords(mc1gt_2,150,147,261).  visible(mc1gt_2,261). happensAt(walking(mc1gt_2),261).
orientation(mc1gt_1,165,262). coords(mc1gt_1,208,181,262).  visible(mc1gt_1,262). happensAt(walking(mc1gt_1),262). orientation(mc1gt_2,146,262). coords(mc1gt_2,151,148,262).  visible(mc1gt_2,262). happensAt(walking(mc1gt_2),262).
orientation(mc1gt_1,168,263). coords(mc1gt_1,208,183,263).  visible(mc1gt_1,263). happensAt(walking(mc1gt_1),263). orientation(mc1gt_2,146,263). coords(mc1gt_2,155,150,263).  visible(mc1gt_2,263). happensAt(walking(mc1gt_2),263).
orientation(mc1gt_1,171,264). coords(mc1gt_1,209,185,264).  visible(mc1gt_1,264). happensAt(walking(mc1gt_1),264). orientation(mc1gt_2,146,264). coords(mc1gt_2,156,151,264).  visible(mc1gt_2,264). happensAt(walking(mc1gt_2),264).
orientation(mc1gt_1,172,265). coords(mc1gt_1,209,186,265).  visible(mc1gt_1,265). happensAt(walking(mc1gt_1),265). orientation(mc1gt_2,148,265). coords(mc1gt_2,157,152,265).  visible(mc1gt_2,265). happensAt(walking(mc1gt_2),265).
orientation(mc1gt_1,174,266). coords(mc1gt_1,209,186,266).  visible(mc1gt_1,266). happensAt(walking(mc1gt_1),266). orientation(mc1gt_2,148,266). coords(mc1gt_2,157,153,266).  visible(mc1gt_2,266). happensAt(walking(mc1gt_2),266).
orientation(mc1gt_1,174,267). coords(mc1gt_1,209,186,267).  visible(mc1gt_1,267). happensAt(walking(mc1gt_1),267). orientation(mc1gt_2,149,267). coords(mc1gt_2,157,154,267).  visible(mc1gt_2,267). happensAt(walking(mc1gt_2),267).
orientation(mc1gt_1,175,268). coords(mc1gt_1,209,187,268).  visible(mc1gt_1,268). happensAt(walking(mc1gt_1),268). orientation(mc1gt_2,149,268). coords(mc1gt_2,158,155,268).  visible(mc1gt_2,268). happensAt(walking(mc1gt_2),268).
orientation(mc1gt_1,172,269). coords(mc1gt_1,209,189,269).  visible(mc1gt_1,269). happensAt(walking(mc1gt_1),269). orientation(mc1gt_2,152,269). coords(mc1gt_2,158,155,269).  visible(mc1gt_2,269). happensAt(walking(mc1gt_2),269).
orientation(mc1gt_1,177,270). coords(mc1gt_1,210,190,270).  visible(mc1gt_1,270). happensAt(walking(mc1gt_1),270). orientation(mc1gt_2,153,270). coords(mc1gt_2,160,156,270).  visible(mc1gt_2,270). happensAt(walking(mc1gt_2),270).
orientation(mc1gt_1,177,271). coords(mc1gt_1,211,192,271).  visible(mc1gt_1,271). happensAt(walking(mc1gt_1),271). orientation(mc1gt_2,154,271). coords(mc1gt_2,164,156,271).  visible(mc1gt_2,271). happensAt(walking(mc1gt_2),271).
orientation(mc1gt_1,163,272). coords(mc1gt_1,212,193,272).  visible(mc1gt_1,272). happensAt(walking(mc1gt_1),272). orientation(mc1gt_2,157,272). coords(mc1gt_2,166,157,272).  visible(mc1gt_2,272). happensAt(walking(mc1gt_2),272).
orientation(mc1gt_1,165,273). coords(mc1gt_1,213,194,273).  visible(mc1gt_1,273). happensAt(walking(mc1gt_1),273). orientation(mc1gt_2,153,273). coords(mc1gt_2,169,159,273).  visible(mc1gt_2,273). happensAt(walking(mc1gt_2),273).
#show holdsAt/2.