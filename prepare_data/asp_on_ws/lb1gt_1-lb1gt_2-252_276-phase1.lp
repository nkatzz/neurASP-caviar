
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

	orientation(lb1gt_1,151,252). coords(lb1gt_1,150,160,252).  visible(lb1gt_1,252). happensAt(walking(lb1gt_1),252). orientation(lb1gt_2,149,252). coords(lb1gt_2,161,130,252).  visible(lb1gt_2,252). happensAt(walking(lb1gt_2),252).
orientation(lb1gt_1,151,253). coords(lb1gt_1,151,160,253).  visible(lb1gt_1,253). happensAt(active(lb1gt_1),253). orientation(lb1gt_2,149,253). coords(lb1gt_2,161,131,253).  visible(lb1gt_2,253). happensAt(walking(lb1gt_2),253).
orientation(lb1gt_1,151,254). coords(lb1gt_1,151,160,254).  visible(lb1gt_1,254). happensAt(active(lb1gt_1),254). orientation(lb1gt_2,149,254). coords(lb1gt_2,161,131,254).  visible(lb1gt_2,254). happensAt(walking(lb1gt_2),254).
orientation(lb1gt_1,151,255). coords(lb1gt_1,151,160,255).  visible(lb1gt_1,255). happensAt(active(lb1gt_1),255). orientation(lb1gt_2,149,255). coords(lb1gt_2,161,131,255).  visible(lb1gt_2,255). happensAt(walking(lb1gt_2),255).
orientation(lb1gt_1,151,256). coords(lb1gt_1,152,160,256).  visible(lb1gt_1,256). happensAt(active(lb1gt_1),256). orientation(lb1gt_2,149,256). coords(lb1gt_2,161,131,256).  visible(lb1gt_2,256). happensAt(walking(lb1gt_2),256).
orientation(lb1gt_1,151,257). coords(lb1gt_1,152,160,257).  visible(lb1gt_1,257). happensAt(active(lb1gt_1),257). orientation(lb1gt_2,149,257). coords(lb1gt_2,161,131,257).  visible(lb1gt_2,257). happensAt(walking(lb1gt_2),257).
orientation(lb1gt_1,151,258). coords(lb1gt_1,152,160,258).  visible(lb1gt_1,258). happensAt(active(lb1gt_1),258). orientation(lb1gt_2,149,258). coords(lb1gt_2,161,131,258).  visible(lb1gt_2,258). happensAt(walking(lb1gt_2),258).
orientation(lb1gt_1,151,259). coords(lb1gt_1,152,160,259).  visible(lb1gt_1,259). happensAt(active(lb1gt_1),259). orientation(lb1gt_2,149,259). coords(lb1gt_2,162,132,259).  visible(lb1gt_2,259). happensAt(walking(lb1gt_2),259).
orientation(lb1gt_1,151,260). coords(lb1gt_1,152,160,260).  visible(lb1gt_1,260). happensAt(active(lb1gt_1),260). orientation(lb1gt_2,149,260). coords(lb1gt_2,163,132,260).  visible(lb1gt_2,260). happensAt(walking(lb1gt_2),260).
orientation(lb1gt_1,151,261). coords(lb1gt_1,153,160,261).  visible(lb1gt_1,261). happensAt(active(lb1gt_1),261). orientation(lb1gt_2,149,261). coords(lb1gt_2,165,132,261).  visible(lb1gt_2,261). happensAt(walking(lb1gt_2),261).
orientation(lb1gt_1,151,262). coords(lb1gt_1,153,160,262).  visible(lb1gt_1,262). happensAt(active(lb1gt_1),262). orientation(lb1gt_2,149,262). coords(lb1gt_2,168,132,262).  visible(lb1gt_2,262). happensAt(walking(lb1gt_2),262).
orientation(lb1gt_1,151,263). coords(lb1gt_1,153,160,263).  visible(lb1gt_1,263). happensAt(active(lb1gt_1),263). orientation(lb1gt_2,149,263). coords(lb1gt_2,170,132,263).  visible(lb1gt_2,263). happensAt(walking(lb1gt_2),263).
orientation(lb1gt_1,151,264). coords(lb1gt_1,153,160,264).  visible(lb1gt_1,264). happensAt(active(lb1gt_1),264). orientation(lb1gt_2,149,264). coords(lb1gt_2,172,134,264).  visible(lb1gt_2,264). happensAt(walking(lb1gt_2),264).
orientation(lb1gt_1,151,265). coords(lb1gt_1,153,160,265).  visible(lb1gt_1,265). happensAt(active(lb1gt_1),265). orientation(lb1gt_2,149,265). coords(lb1gt_2,172,134,265).  visible(lb1gt_2,265). happensAt(walking(lb1gt_2),265).
orientation(lb1gt_1,151,266). coords(lb1gt_1,153,160,266).  visible(lb1gt_1,266). happensAt(active(lb1gt_1),266). orientation(lb1gt_2,149,266). coords(lb1gt_2,173,134,266).  visible(lb1gt_2,266). happensAt(walking(lb1gt_2),266).
orientation(lb1gt_1,151,267). coords(lb1gt_1,153,160,267).  visible(lb1gt_1,267). happensAt(active(lb1gt_1),267). orientation(lb1gt_2,149,267). coords(lb1gt_2,175,135,267).  visible(lb1gt_2,267). happensAt(walking(lb1gt_2),267).
orientation(lb1gt_1,151,268). coords(lb1gt_1,153,160,268).  visible(lb1gt_1,268). happensAt(active(lb1gt_1),268). orientation(lb1gt_2,149,268). coords(lb1gt_2,175,135,268).  visible(lb1gt_2,268). happensAt(walking(lb1gt_2),268).
orientation(lb1gt_1,151,269). coords(lb1gt_1,154,160,269).  visible(lb1gt_1,269). happensAt(active(lb1gt_1),269). orientation(lb1gt_2,149,269). coords(lb1gt_2,176,135,269).  visible(lb1gt_2,269). happensAt(walking(lb1gt_2),269).
orientation(lb1gt_1,151,270). coords(lb1gt_1,154,160,270).  visible(lb1gt_1,270). happensAt(active(lb1gt_1),270). orientation(lb1gt_2,149,270). coords(lb1gt_2,176,135,270).  visible(lb1gt_2,270). happensAt(walking(lb1gt_2),270).
orientation(lb1gt_1,151,271). coords(lb1gt_1,154,159,271).  visible(lb1gt_1,271). happensAt(active(lb1gt_1),271). orientation(lb1gt_2,149,271). coords(lb1gt_2,177,135,271).  visible(lb1gt_2,271). happensAt(walking(lb1gt_2),271).
orientation(lb1gt_1,151,272). coords(lb1gt_1,154,159,272).  visible(lb1gt_1,272). happensAt(active(lb1gt_1),272). orientation(lb1gt_2,146,272). coords(lb1gt_2,178,136,272).  visible(lb1gt_2,272). happensAt(walking(lb1gt_2),272).
orientation(lb1gt_1,151,273). coords(lb1gt_1,154,159,273).  visible(lb1gt_1,273). happensAt(active(lb1gt_1),273). orientation(lb1gt_2,146,273). coords(lb1gt_2,178,136,273).  visible(lb1gt_2,273). happensAt(walking(lb1gt_2),273).
orientation(lb1gt_1,151,274). coords(lb1gt_1,154,159,274).  visible(lb1gt_1,274). happensAt(active(lb1gt_1),274). orientation(lb1gt_2,146,274). coords(lb1gt_2,178,137,274).  visible(lb1gt_2,274). happensAt(walking(lb1gt_2),274).
orientation(lb1gt_1,151,275). coords(lb1gt_1,154,159,275).  visible(lb1gt_1,275). happensAt(active(lb1gt_1),275). orientation(lb1gt_2,146,275). coords(lb1gt_2,179,138,275).  visible(lb1gt_2,275). happensAt(walking(lb1gt_2),275).
orientation(lb1gt_1,151,276). coords(lb1gt_1,154,159,276).  visible(lb1gt_1,276). happensAt(active(lb1gt_1),276). orientation(lb1gt_2,146,276). coords(lb1gt_2,180,140,276).  visible(lb1gt_2,276). happensAt(walking(lb1gt_2),276).
#show holdsAt/2.