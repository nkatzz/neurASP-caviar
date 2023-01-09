
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
orientation(mc1gt_0,152,269). coords(mc1gt_0,182,182,269).  visible(mc1gt_0,269). happensAt(walking(mc1gt_0),269). orientation(mc1gt_1,172,269). coords(mc1gt_1,209,189,269).  visible(mc1gt_1,269). happensAt(walking(mc1gt_1),269).
orientation(mc1gt_0,152,270). coords(mc1gt_0,183,185,270).  visible(mc1gt_0,270). happensAt(walking(mc1gt_0),270). orientation(mc1gt_1,177,270). coords(mc1gt_1,210,190,270).  visible(mc1gt_1,270). happensAt(walking(mc1gt_1),270).
orientation(mc1gt_0,152,271). coords(mc1gt_0,183,188,271).  visible(mc1gt_0,271). happensAt(walking(mc1gt_0),271). orientation(mc1gt_1,177,271). coords(mc1gt_1,211,192,271).  visible(mc1gt_1,271). happensAt(walking(mc1gt_1),271).
orientation(mc1gt_0,156,272). coords(mc1gt_0,184,191,272).  visible(mc1gt_0,272). happensAt(walking(mc1gt_0),272). orientation(mc1gt_1,163,272). coords(mc1gt_1,212,193,272).  visible(mc1gt_1,272). happensAt(walking(mc1gt_1),272).
orientation(mc1gt_0,156,273). coords(mc1gt_0,184,192,273).  visible(mc1gt_0,273). happensAt(walking(mc1gt_0),273). orientation(mc1gt_1,165,273). coords(mc1gt_1,213,194,273).  visible(mc1gt_1,273). happensAt(walking(mc1gt_1),273).
orientation(mc1gt_0,156,274). coords(mc1gt_0,185,194,274).  visible(mc1gt_0,274). happensAt(walking(mc1gt_0),274). orientation(mc1gt_1,168,274). coords(mc1gt_1,214,196,274).  visible(mc1gt_1,274). happensAt(walking(mc1gt_1),274).
orientation(mc1gt_0,156,275). coords(mc1gt_0,186,195,275).  visible(mc1gt_0,275). happensAt(walking(mc1gt_0),275). orientation(mc1gt_1,171,275). coords(mc1gt_1,215,200,275).  visible(mc1gt_1,275). happensAt(walking(mc1gt_1),275).
orientation(mc1gt_0,156,276). coords(mc1gt_0,186,195,276).  visible(mc1gt_0,276). happensAt(walking(mc1gt_0),276). orientation(mc1gt_1,170,276). coords(mc1gt_1,217,204,276).  visible(mc1gt_1,276). happensAt(walking(mc1gt_1),276).
orientation(mc1gt_0,156,277). coords(mc1gt_0,186,196,277).  visible(mc1gt_0,277). happensAt(walking(mc1gt_0),277). orientation(mc1gt_1,169,277). coords(mc1gt_1,218,208,277).  visible(mc1gt_1,277). happensAt(walking(mc1gt_1),277).
orientation(mc1gt_0,163,278). coords(mc1gt_0,186,196,278).  visible(mc1gt_0,278). happensAt(walking(mc1gt_0),278). orientation(mc1gt_1,175,278). coords(mc1gt_1,218,210,278).  visible(mc1gt_1,278). happensAt(walking(mc1gt_1),278).
orientation(mc1gt_0,163,279). coords(mc1gt_0,186,198,279).  visible(mc1gt_0,279). happensAt(walking(mc1gt_0),279). orientation(mc1gt_1,175,279). coords(mc1gt_1,219,211,279).  visible(mc1gt_1,279). happensAt(walking(mc1gt_1),279).
orientation(mc1gt_0,163,280). coords(mc1gt_0,186,199,280).  visible(mc1gt_0,280). happensAt(walking(mc1gt_0),280). orientation(mc1gt_1,174,280). coords(mc1gt_1,219,211,280).  visible(mc1gt_1,280). happensAt(walking(mc1gt_1),280).
#show holdsAt/2.