
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

	orientation(ms3ggt_0,157,266). coords(ms3ggt_0,63,151,266).  visible(ms3ggt_0,266). happensAt(walking(ms3ggt_0),266). orientation(ms3ggt_1,158,266). coords(ms3ggt_1,62,165,266).  visible(ms3ggt_1,266). happensAt(walking(ms3ggt_1),266).
orientation(ms3ggt_0,157,267). coords(ms3ggt_0,63,151,267).  visible(ms3ggt_0,267). happensAt(walking(ms3ggt_0),267). orientation(ms3ggt_1,158,267). coords(ms3ggt_1,62,165,267).  visible(ms3ggt_1,267). happensAt(walking(ms3ggt_1),267).
orientation(ms3ggt_0,157,268). coords(ms3ggt_0,63,151,268).  visible(ms3ggt_0,268). happensAt(walking(ms3ggt_0),268). orientation(ms3ggt_1,158,268). coords(ms3ggt_1,62,166,268).  visible(ms3ggt_1,268). happensAt(walking(ms3ggt_1),268).
orientation(ms3ggt_0,157,269). coords(ms3ggt_0,63,150,269).  visible(ms3ggt_0,269). happensAt(walking(ms3ggt_0),269). orientation(ms3ggt_1,158,269). coords(ms3ggt_1,62,166,269).  visible(ms3ggt_1,269). happensAt(walking(ms3ggt_1),269).
orientation(ms3ggt_0,157,270). coords(ms3ggt_0,63,150,270).  visible(ms3ggt_0,270). happensAt(walking(ms3ggt_0),270). orientation(ms3ggt_1,158,270). coords(ms3ggt_1,62,166,270).  visible(ms3ggt_1,270). happensAt(walking(ms3ggt_1),270).
orientation(ms3ggt_0,157,271). coords(ms3ggt_0,63,150,271).  visible(ms3ggt_0,271). happensAt(walking(ms3ggt_0),271). orientation(ms3ggt_1,158,271). coords(ms3ggt_1,62,166,271).  visible(ms3ggt_1,271). happensAt(walking(ms3ggt_1),271).
orientation(ms3ggt_0,157,272). coords(ms3ggt_0,65,150,272).  visible(ms3ggt_0,272). happensAt(walking(ms3ggt_0),272). orientation(ms3ggt_1,158,272). coords(ms3ggt_1,62,166,272).  visible(ms3ggt_1,272). happensAt(walking(ms3ggt_1),272).
orientation(ms3ggt_0,157,273). coords(ms3ggt_0,66,150,273).  visible(ms3ggt_0,273). happensAt(walking(ms3ggt_0),273). orientation(ms3ggt_1,158,273). coords(ms3ggt_1,63,165,273).  visible(ms3ggt_1,273). happensAt(walking(ms3ggt_1),273).
orientation(ms3ggt_0,161,274). coords(ms3ggt_0,67,150,274).  visible(ms3ggt_0,274). happensAt(walking(ms3ggt_0),274). orientation(ms3ggt_1,158,274). coords(ms3ggt_1,63,164,274).  visible(ms3ggt_1,274). happensAt(walking(ms3ggt_1),274).
orientation(ms3ggt_0,161,275). coords(ms3ggt_0,67,150,275).  visible(ms3ggt_0,275). happensAt(walking(ms3ggt_0),275). orientation(ms3ggt_1,158,275). coords(ms3ggt_1,63,164,275).  visible(ms3ggt_1,275). happensAt(walking(ms3ggt_1),275).
orientation(ms3ggt_0,161,276). coords(ms3ggt_0,68,150,276).  visible(ms3ggt_0,276). happensAt(walking(ms3ggt_0),276). orientation(ms3ggt_1,158,276). coords(ms3ggt_1,63,164,276).  visible(ms3ggt_1,276). happensAt(walking(ms3ggt_1),276).
orientation(ms3ggt_0,161,277). coords(ms3ggt_0,68,150,277).  visible(ms3ggt_0,277). happensAt(walking(ms3ggt_0),277). orientation(ms3ggt_1,158,277). coords(ms3ggt_1,66,164,277).  visible(ms3ggt_1,277). happensAt(walking(ms3ggt_1),277).
orientation(ms3ggt_0,161,278). coords(ms3ggt_0,68,149,278).  visible(ms3ggt_0,278). happensAt(walking(ms3ggt_0),278). orientation(ms3ggt_1,158,278). coords(ms3ggt_1,66,164,278).  visible(ms3ggt_1,278). happensAt(walking(ms3ggt_1),278).
orientation(ms3ggt_0,161,279). coords(ms3ggt_0,69,149,279).  visible(ms3ggt_0,279). happensAt(walking(ms3ggt_0),279). orientation(ms3ggt_1,158,279). coords(ms3ggt_1,68,164,279).  visible(ms3ggt_1,279). happensAt(walking(ms3ggt_1),279).
orientation(ms3ggt_0,161,280). coords(ms3ggt_0,69,149,280).  visible(ms3ggt_0,280). happensAt(walking(ms3ggt_0),280). orientation(ms3ggt_1,158,280). coords(ms3ggt_1,68,164,280).  visible(ms3ggt_1,280). happensAt(walking(ms3ggt_1),280).
orientation(ms3ggt_0,161,281). coords(ms3ggt_0,69,149,281).  visible(ms3ggt_0,281). happensAt(walking(ms3ggt_0),281). orientation(ms3ggt_1,158,281). coords(ms3ggt_1,68,164,281).  visible(ms3ggt_1,281). happensAt(walking(ms3ggt_1),281).
orientation(ms3ggt_0,153,282). coords(ms3ggt_0,69,148,282).  visible(ms3ggt_0,282). happensAt(walking(ms3ggt_0),282). orientation(ms3ggt_1,158,282). coords(ms3ggt_1,69,164,282).  visible(ms3ggt_1,282). happensAt(walking(ms3ggt_1),282).
orientation(ms3ggt_0,153,283). coords(ms3ggt_0,69,148,283).  visible(ms3ggt_0,283). happensAt(walking(ms3ggt_0),283). orientation(ms3ggt_1,158,283). coords(ms3ggt_1,69,164,283).  visible(ms3ggt_1,283). happensAt(walking(ms3ggt_1),283).
orientation(ms3ggt_0,153,284). coords(ms3ggt_0,69,148,284).  visible(ms3ggt_0,284). happensAt(walking(ms3ggt_0),284). orientation(ms3ggt_1,158,284). coords(ms3ggt_1,69,164,284).  visible(ms3ggt_1,284). happensAt(walking(ms3ggt_1),284).
orientation(ms3ggt_0,153,285). coords(ms3ggt_0,69,148,285).  visible(ms3ggt_0,285). happensAt(walking(ms3ggt_0),285). orientation(ms3ggt_1,158,285). coords(ms3ggt_1,70,164,285).  visible(ms3ggt_1,285). happensAt(walking(ms3ggt_1),285).
orientation(ms3ggt_0,153,286). coords(ms3ggt_0,69,146,286).  visible(ms3ggt_0,286). happensAt(walking(ms3ggt_0),286). orientation(ms3ggt_1,158,286). coords(ms3ggt_1,70,164,286).  visible(ms3ggt_1,286). happensAt(walking(ms3ggt_1),286).
orientation(ms3ggt_0,153,287). coords(ms3ggt_0,73,146,287).  visible(ms3ggt_0,287). happensAt(walking(ms3ggt_0),287). orientation(ms3ggt_1,158,287). coords(ms3ggt_1,70,164,287).  visible(ms3ggt_1,287). happensAt(walking(ms3ggt_1),287).
orientation(ms3ggt_0,153,288). coords(ms3ggt_0,74,146,288).  visible(ms3ggt_0,288). happensAt(walking(ms3ggt_0),288). orientation(ms3ggt_1,158,288). coords(ms3ggt_1,70,164,288).  visible(ms3ggt_1,288). happensAt(walking(ms3ggt_1),288).
orientation(ms3ggt_0,153,289). coords(ms3ggt_0,74,146,289).  visible(ms3ggt_0,289). happensAt(walking(ms3ggt_0),289). orientation(ms3ggt_1,158,289). coords(ms3ggt_1,71,164,289).  visible(ms3ggt_1,289). happensAt(walking(ms3ggt_1),289).
orientation(ms3ggt_0,159,290). coords(ms3ggt_0,76,146,290).  visible(ms3ggt_0,290). happensAt(walking(ms3ggt_0),290). orientation(ms3ggt_1,158,290). coords(ms3ggt_1,72,164,290).  visible(ms3ggt_1,290). happensAt(walking(ms3ggt_1),290).
#show holdsAt/2.