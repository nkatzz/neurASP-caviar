
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
orientation(mc1gt_1,168,274). coords(mc1gt_1,214,196,274).  visible(mc1gt_1,274). happensAt(walking(mc1gt_1),274). orientation(mc1gt_2,152,274). coords(mc1gt_2,171,159,274).  visible(mc1gt_2,274). happensAt(walking(mc1gt_2),274).
orientation(mc1gt_1,171,275). coords(mc1gt_1,215,200,275).  visible(mc1gt_1,275). happensAt(walking(mc1gt_1),275). orientation(mc1gt_2,152,275). coords(mc1gt_2,172,161,275).  visible(mc1gt_2,275). happensAt(walking(mc1gt_2),275).
orientation(mc1gt_1,170,276). coords(mc1gt_1,217,204,276).  visible(mc1gt_1,276). happensAt(walking(mc1gt_1),276). orientation(mc1gt_2,160,276). coords(mc1gt_2,174,162,276).  visible(mc1gt_2,276). happensAt(walking(mc1gt_2),276).
orientation(mc1gt_1,169,277). coords(mc1gt_1,218,208,277).  visible(mc1gt_1,277). happensAt(walking(mc1gt_1),277). orientation(mc1gt_2,161,277). coords(mc1gt_2,176,162,277).  visible(mc1gt_2,277). happensAt(walking(mc1gt_2),277).
orientation(mc1gt_1,175,278). coords(mc1gt_1,218,210,278).  visible(mc1gt_1,278). happensAt(walking(mc1gt_1),278). orientation(mc1gt_2,164,278). coords(mc1gt_2,176,163,278).  visible(mc1gt_2,278). happensAt(walking(mc1gt_2),278).
orientation(mc1gt_1,175,279). coords(mc1gt_1,219,211,279).  visible(mc1gt_1,279). happensAt(walking(mc1gt_1),279). orientation(mc1gt_2,164,279). coords(mc1gt_2,177,164,279).  visible(mc1gt_2,279). happensAt(walking(mc1gt_2),279).
orientation(mc1gt_1,174,280). coords(mc1gt_1,219,211,280).  visible(mc1gt_1,280). happensAt(walking(mc1gt_1),280). orientation(mc1gt_2,152,280). coords(mc1gt_2,177,164,280).  visible(mc1gt_2,280). happensAt(walking(mc1gt_2),280).
orientation(mc1gt_1,178,281). coords(mc1gt_1,220,211,281).  visible(mc1gt_1,281). happensAt(walking(mc1gt_1),281). orientation(mc1gt_2,152,281). coords(mc1gt_2,178,164,281).  visible(mc1gt_2,281). happensAt(walking(mc1gt_2),281).
orientation(mc1gt_1,1,282). coords(mc1gt_1,221,211,282).  visible(mc1gt_1,282). happensAt(walking(mc1gt_1),282). orientation(mc1gt_2,152,282). coords(mc1gt_2,178,165,282).  visible(mc1gt_2,282). happensAt(walking(mc1gt_2),282).
orientation(mc1gt_1,6,283). coords(mc1gt_1,221,212,283).  visible(mc1gt_1,283). happensAt(walking(mc1gt_1),283). orientation(mc1gt_2,152,283). coords(mc1gt_2,179,166,283).  visible(mc1gt_2,283). happensAt(walking(mc1gt_2),283).
orientation(mc1gt_1,4,284). coords(mc1gt_1,222,213,284).  visible(mc1gt_1,284). happensAt(walking(mc1gt_1),284). orientation(mc1gt_2,152,284). coords(mc1gt_2,180,168,284).  visible(mc1gt_2,284). happensAt(walking(mc1gt_2),284).
orientation(mc1gt_1,4,285). coords(mc1gt_1,222,215,285).  visible(mc1gt_1,285). happensAt(walking(mc1gt_1),285). orientation(mc1gt_2,152,285). coords(mc1gt_2,180,170,285).  visible(mc1gt_2,285). happensAt(walking(mc1gt_2),285).
#show holdsAt/2.