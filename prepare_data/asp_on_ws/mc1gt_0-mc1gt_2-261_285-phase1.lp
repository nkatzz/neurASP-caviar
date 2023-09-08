
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
orientation(mc1gt_0,156,274). coords(mc1gt_0,185,194,274).  visible(mc1gt_0,274). happensAt(walking(mc1gt_0),274). orientation(mc1gt_2,152,274). coords(mc1gt_2,171,159,274).  visible(mc1gt_2,274). happensAt(walking(mc1gt_2),274).
orientation(mc1gt_0,156,275). coords(mc1gt_0,186,195,275).  visible(mc1gt_0,275). happensAt(walking(mc1gt_0),275). orientation(mc1gt_2,152,275). coords(mc1gt_2,172,161,275).  visible(mc1gt_2,275). happensAt(walking(mc1gt_2),275).
orientation(mc1gt_0,156,276). coords(mc1gt_0,186,195,276).  visible(mc1gt_0,276). happensAt(walking(mc1gt_0),276). orientation(mc1gt_2,160,276). coords(mc1gt_2,174,162,276).  visible(mc1gt_2,276). happensAt(walking(mc1gt_2),276).
orientation(mc1gt_0,156,277). coords(mc1gt_0,186,196,277).  visible(mc1gt_0,277). happensAt(walking(mc1gt_0),277). orientation(mc1gt_2,161,277). coords(mc1gt_2,176,162,277).  visible(mc1gt_2,277). happensAt(walking(mc1gt_2),277).
orientation(mc1gt_0,163,278). coords(mc1gt_0,186,196,278).  visible(mc1gt_0,278). happensAt(walking(mc1gt_0),278). orientation(mc1gt_2,164,278). coords(mc1gt_2,176,163,278).  visible(mc1gt_2,278). happensAt(walking(mc1gt_2),278).
orientation(mc1gt_0,163,279). coords(mc1gt_0,186,198,279).  visible(mc1gt_0,279). happensAt(walking(mc1gt_0),279). orientation(mc1gt_2,164,279). coords(mc1gt_2,177,164,279).  visible(mc1gt_2,279). happensAt(walking(mc1gt_2),279).
orientation(mc1gt_0,163,280). coords(mc1gt_0,186,199,280).  visible(mc1gt_0,280). happensAt(walking(mc1gt_0),280). orientation(mc1gt_2,152,280). coords(mc1gt_2,177,164,280).  visible(mc1gt_2,280). happensAt(walking(mc1gt_2),280).
orientation(mc1gt_0,163,281). coords(mc1gt_0,187,202,281).  visible(mc1gt_0,281). happensAt(walking(mc1gt_0),281). orientation(mc1gt_2,152,281). coords(mc1gt_2,178,164,281).  visible(mc1gt_2,281). happensAt(walking(mc1gt_2),281).
orientation(mc1gt_0,163,282). coords(mc1gt_0,189,204,282).  visible(mc1gt_0,282). happensAt(walking(mc1gt_0),282). orientation(mc1gt_2,152,282). coords(mc1gt_2,178,165,282).  visible(mc1gt_2,282). happensAt(walking(mc1gt_2),282).
orientation(mc1gt_0,163,283). coords(mc1gt_0,190,205,283).  visible(mc1gt_0,283). happensAt(walking(mc1gt_0),283). orientation(mc1gt_2,152,283). coords(mc1gt_2,179,166,283).  visible(mc1gt_2,283). happensAt(walking(mc1gt_2),283).
orientation(mc1gt_0,168,284). coords(mc1gt_0,190,207,284).  visible(mc1gt_0,284). happensAt(walking(mc1gt_0),284). orientation(mc1gt_2,152,284). coords(mc1gt_2,180,168,284).  visible(mc1gt_2,284). happensAt(walking(mc1gt_2),284).
orientation(mc1gt_0,168,285). coords(mc1gt_0,189,209,285).  visible(mc1gt_0,285). happensAt(walking(mc1gt_0),285). orientation(mc1gt_2,152,285). coords(mc1gt_2,180,170,285).  visible(mc1gt_2,285). happensAt(walking(mc1gt_2),285).
#show holdsAt/2.