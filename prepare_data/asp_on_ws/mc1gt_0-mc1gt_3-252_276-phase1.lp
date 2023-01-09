
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
orientation(mc1gt_0,152,265). coords(mc1gt_0,182,176,265).  visible(mc1gt_0,265). happensAt(walking(mc1gt_0),265). orientation(mc1gt_3,131,265). coords(mc1gt_3,217,114,265).  visible(mc1gt_3,265). happensAt(walking(mc1gt_3),265).
orientation(mc1gt_0,152,266). coords(mc1gt_0,182,177,266).  visible(mc1gt_0,266). happensAt(walking(mc1gt_0),266). orientation(mc1gt_3,131,266). coords(mc1gt_3,220,117,266).  visible(mc1gt_3,266). happensAt(walking(mc1gt_3),266).
orientation(mc1gt_0,152,267). coords(mc1gt_0,182,178,267).  visible(mc1gt_0,267). happensAt(walking(mc1gt_0),267). orientation(mc1gt_3,132,267). coords(mc1gt_3,223,119,267).  visible(mc1gt_3,267). happensAt(walking(mc1gt_3),267).
orientation(mc1gt_0,152,268). coords(mc1gt_0,182,179,268).  visible(mc1gt_0,268). happensAt(walking(mc1gt_0),268). orientation(mc1gt_3,132,268). coords(mc1gt_3,226,121,268).  visible(mc1gt_3,268). happensAt(walking(mc1gt_3),268).
orientation(mc1gt_0,152,269). coords(mc1gt_0,182,182,269).  visible(mc1gt_0,269). happensAt(walking(mc1gt_0),269). orientation(mc1gt_3,132,269). coords(mc1gt_3,228,123,269).  visible(mc1gt_3,269). happensAt(walking(mc1gt_3),269).
orientation(mc1gt_0,152,270). coords(mc1gt_0,183,185,270).  visible(mc1gt_0,270). happensAt(walking(mc1gt_0),270). orientation(mc1gt_3,132,270). coords(mc1gt_3,230,124,270).  visible(mc1gt_3,270). happensAt(walking(mc1gt_3),270).
orientation(mc1gt_0,152,271). coords(mc1gt_0,183,188,271).  visible(mc1gt_0,271). happensAt(walking(mc1gt_0),271). orientation(mc1gt_3,132,271). coords(mc1gt_3,231,125,271).  visible(mc1gt_3,271). happensAt(walking(mc1gt_3),271).
orientation(mc1gt_0,156,272). coords(mc1gt_0,184,191,272).  visible(mc1gt_0,272). happensAt(walking(mc1gt_0),272). orientation(mc1gt_3,132,272). coords(mc1gt_3,232,126,272).  visible(mc1gt_3,272). happensAt(walking(mc1gt_3),272).
orientation(mc1gt_0,156,273). coords(mc1gt_0,184,192,273).  visible(mc1gt_0,273). happensAt(walking(mc1gt_0),273). orientation(mc1gt_3,132,273). coords(mc1gt_3,233,126,273).  visible(mc1gt_3,273). happensAt(walking(mc1gt_3),273).
orientation(mc1gt_0,156,274). coords(mc1gt_0,185,194,274).  visible(mc1gt_0,274). happensAt(walking(mc1gt_0),274). orientation(mc1gt_3,135,274). coords(mc1gt_3,234,127,274).  visible(mc1gt_3,274). happensAt(walking(mc1gt_3),274).
orientation(mc1gt_0,156,275). coords(mc1gt_0,186,195,275).  visible(mc1gt_0,275). happensAt(walking(mc1gt_0),275). orientation(mc1gt_3,135,275). coords(mc1gt_3,234,128,275).  visible(mc1gt_3,275). happensAt(walking(mc1gt_3),275).
orientation(mc1gt_0,156,276). coords(mc1gt_0,186,195,276).  visible(mc1gt_0,276). happensAt(walking(mc1gt_0),276). orientation(mc1gt_3,135,276). coords(mc1gt_3,235,128,276).  visible(mc1gt_3,276). happensAt(walking(mc1gt_3),276).
#show holdsAt/2.