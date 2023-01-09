
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

	orientation(fra1gt_0,39,69). coords(fra1gt_0,264,254,69).  visible(fra1gt_0,69). happensAt(walking(fra1gt_0),69). orientation(fra1gt_7,136,69). coords(fra1gt_7,144,39,69).  visible(fra1gt_7,69). happensAt(inactive(fra1gt_7),69).
orientation(fra1gt_0,39,70). coords(fra1gt_0,266,255,70).  visible(fra1gt_0,70). happensAt(walking(fra1gt_0),70). orientation(fra1gt_7,136,70). coords(fra1gt_7,144,39,70).  visible(fra1gt_7,70). happensAt(inactive(fra1gt_7),70).
orientation(fra1gt_0,39,71). coords(fra1gt_0,267,257,71).  visible(fra1gt_0,71). happensAt(walking(fra1gt_0),71). orientation(fra1gt_7,136,71). coords(fra1gt_7,144,39,71).  visible(fra1gt_7,71). happensAt(inactive(fra1gt_7),71).
orientation(fra1gt_0,39,72). coords(fra1gt_0,269,258,72).  visible(fra1gt_0,72). happensAt(walking(fra1gt_0),72). orientation(fra1gt_7,136,72). coords(fra1gt_7,144,39,72).  visible(fra1gt_7,72). happensAt(inactive(fra1gt_7),72).
orientation(fra1gt_0,46,73). coords(fra1gt_0,270,260,73).  visible(fra1gt_0,73). happensAt(walking(fra1gt_0),73). orientation(fra1gt_7,136,73). coords(fra1gt_7,145,39,73).  visible(fra1gt_7,73). happensAt(inactive(fra1gt_7),73).
orientation(fra1gt_0,46,74). coords(fra1gt_0,271,262,74).  visible(fra1gt_0,74). happensAt(walking(fra1gt_0),74). orientation(fra1gt_7,136,74). coords(fra1gt_7,145,39,74).  visible(fra1gt_7,74). happensAt(inactive(fra1gt_7),74).
orientation(fra1gt_0,46,75). coords(fra1gt_0,272,264,75).  visible(fra1gt_0,75). happensAt(walking(fra1gt_0),75). orientation(fra1gt_7,136,75). coords(fra1gt_7,145,39,75).  visible(fra1gt_7,75). happensAt(inactive(fra1gt_7),75).
orientation(fra1gt_0,46,76). coords(fra1gt_0,273,266,76).  visible(fra1gt_0,76). happensAt(walking(fra1gt_0),76). orientation(fra1gt_7,136,76). coords(fra1gt_7,145,39,76).  visible(fra1gt_7,76). happensAt(inactive(fra1gt_7),76).
orientation(fra1gt_0,46,77). coords(fra1gt_0,274,268,77).  visible(fra1gt_0,77). happensAt(walking(fra1gt_0),77). orientation(fra1gt_7,136,77). coords(fra1gt_7,146,39,77).  visible(fra1gt_7,77). happensAt(inactive(fra1gt_7),77).
orientation(fra1gt_0,57,78). coords(fra1gt_0,274,270,78).  visible(fra1gt_0,78). happensAt(walking(fra1gt_0),78). orientation(fra1gt_7,136,78). coords(fra1gt_7,146,39,78).  visible(fra1gt_7,78). happensAt(inactive(fra1gt_7),78).
orientation(fra1gt_0,57,79). coords(fra1gt_0,275,273,79).  visible(fra1gt_0,79). happensAt(walking(fra1gt_0),79). orientation(fra1gt_7,136,79). coords(fra1gt_7,146,39,79).  visible(fra1gt_7,79). happensAt(walking(fra1gt_7),79).
orientation(fra1gt_0,57,80). coords(fra1gt_0,275,273,80).  visible(fra1gt_0,80). happensAt(walking(fra1gt_0),80). orientation(fra1gt_7,136,80). coords(fra1gt_7,146,39,80).  visible(fra1gt_7,80). happensAt(walking(fra1gt_7),80).
orientation(fra1gt_0,57,81). coords(fra1gt_0,276,273,81).  visible(fra1gt_0,81). happensAt(walking(fra1gt_0),81). orientation(fra1gt_7,136,81). coords(fra1gt_7,146,39,81).  visible(fra1gt_7,81). happensAt(walking(fra1gt_7),81).
orientation(fra1gt_0,57,82). coords(fra1gt_0,276,273,82).  visible(fra1gt_0,82). happensAt(walking(fra1gt_0),82). orientation(fra1gt_7,136,82). coords(fra1gt_7,146,39,82).  visible(fra1gt_7,82). happensAt(walking(fra1gt_7),82).
orientation(fra1gt_0,57,83). coords(fra1gt_0,278,273,83).  visible(fra1gt_0,83). happensAt(walking(fra1gt_0),83). orientation(fra1gt_7,136,83). coords(fra1gt_7,146,39,83).  visible(fra1gt_7,83). happensAt(walking(fra1gt_7),83).
orientation(fra1gt_0,57,84). coords(fra1gt_0,280,274,84).  visible(fra1gt_0,84). happensAt(walking(fra1gt_0),84). orientation(fra1gt_7,136,84). coords(fra1gt_7,146,39,84).  visible(fra1gt_7,84). happensAt(walking(fra1gt_7),84).
orientation(fra1gt_0,57,85). coords(fra1gt_0,283,274,85).  visible(fra1gt_0,85). happensAt(walking(fra1gt_0),85). orientation(fra1gt_7,136,85). coords(fra1gt_7,146,37,85).  visible(fra1gt_7,85). happensAt(walking(fra1gt_7),85).
orientation(fra1gt_0,76,86). coords(fra1gt_0,285,274,86).  visible(fra1gt_0,86). happensAt(walking(fra1gt_0),86). orientation(fra1gt_7,136,86). coords(fra1gt_7,146,37,86).  visible(fra1gt_7,86). happensAt(walking(fra1gt_7),86).
orientation(fra1gt_0,76,87). coords(fra1gt_0,287,274,87).  visible(fra1gt_0,87). happensAt(walking(fra1gt_0),87). orientation(fra1gt_7,136,87). coords(fra1gt_7,147,36,87).  visible(fra1gt_7,87). happensAt(walking(fra1gt_7),87).
orientation(fra1gt_0,76,88). coords(fra1gt_0,290,276,88).  visible(fra1gt_0,88). happensAt(walking(fra1gt_0),88). orientation(fra1gt_7,136,88). coords(fra1gt_7,147,36,88).  visible(fra1gt_7,88). happensAt(walking(fra1gt_7),88).
orientation(fra1gt_0,76,89). coords(fra1gt_0,292,277,89).  visible(fra1gt_0,89). happensAt(walking(fra1gt_0),89). orientation(fra1gt_7,136,89). coords(fra1gt_7,149,36,89).  visible(fra1gt_7,89). happensAt(walking(fra1gt_7),89).
orientation(fra1gt_0,76,90). coords(fra1gt_0,293,279,90).  visible(fra1gt_0,90). happensAt(walking(fra1gt_0),90). orientation(fra1gt_7,136,90). coords(fra1gt_7,151,35,90).  visible(fra1gt_7,90). happensAt(walking(fra1gt_7),90).
orientation(fra1gt_0,76,91). coords(fra1gt_0,294,281,91).  visible(fra1gt_0,91). happensAt(walking(fra1gt_0),91). orientation(fra1gt_7,136,91). coords(fra1gt_7,152,34,91).  visible(fra1gt_7,91). happensAt(walking(fra1gt_7),91).
orientation(fra1gt_0,76,92). coords(fra1gt_0,297,283,92).  visible(fra1gt_0,92). happensAt(walking(fra1gt_0),92). orientation(fra1gt_7,136,92). coords(fra1gt_7,152,34,92).  visible(fra1gt_7,92). happensAt(walking(fra1gt_7),92).
orientation(fra1gt_0,76,93). coords(fra1gt_0,297,286,93).  happensAt(disappear(fra1gt_0),93). happensAt(walking(fra1gt_0),93). orientation(fra1gt_7,136,93). coords(fra1gt_7,152,33,93).  visible(fra1gt_7,93). happensAt(walking(fra1gt_7),93).
#show holdsAt/2.