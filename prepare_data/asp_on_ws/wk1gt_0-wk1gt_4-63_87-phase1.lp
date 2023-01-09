
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

	orientation(wk1gt_0,143,63). coords(wk1gt_0,223,192,63).  visible(wk1gt_0,63). happensAt(walking(wk1gt_0),63). orientation(wk1gt_4,0,63). coords(wk1gt_4,35,204,63).  visible(wk1gt_4,63). happensAt(walking(wk1gt_4),63).
orientation(wk1gt_0,143,64). coords(wk1gt_0,223,189,64).  visible(wk1gt_0,64). happensAt(walking(wk1gt_0),64). orientation(wk1gt_4,0,64). coords(wk1gt_4,35,204,64).  visible(wk1gt_4,64). happensAt(walking(wk1gt_4),64).
orientation(wk1gt_0,143,65). coords(wk1gt_0,222,186,65).  visible(wk1gt_0,65). happensAt(walking(wk1gt_0),65). orientation(wk1gt_4,0,65). coords(wk1gt_4,35,203,65).  visible(wk1gt_4,65). happensAt(walking(wk1gt_4),65).
orientation(wk1gt_0,140,66). coords(wk1gt_0,222,183,66).  visible(wk1gt_0,66). happensAt(walking(wk1gt_0),66). orientation(wk1gt_4,0,66). coords(wk1gt_4,35,203,66).  visible(wk1gt_4,66). happensAt(walking(wk1gt_4),66).
orientation(wk1gt_0,140,67). coords(wk1gt_0,222,181,67).  visible(wk1gt_0,67). happensAt(walking(wk1gt_0),67). orientation(wk1gt_4,0,67). coords(wk1gt_4,36,203,67).  visible(wk1gt_4,67). happensAt(walking(wk1gt_4),67).
orientation(wk1gt_0,138,68). coords(wk1gt_0,221,179,68).  visible(wk1gt_0,68). happensAt(walking(wk1gt_0),68). orientation(wk1gt_4,0,68). coords(wk1gt_4,36,202,68).  visible(wk1gt_4,68). happensAt(walking(wk1gt_4),68).
orientation(wk1gt_0,136,69). coords(wk1gt_0,220,177,69).  visible(wk1gt_0,69). happensAt(walking(wk1gt_0),69). orientation(wk1gt_4,0,69). coords(wk1gt_4,36,201,69).  visible(wk1gt_4,69). happensAt(walking(wk1gt_4),69).
orientation(wk1gt_0,136,70). coords(wk1gt_0,220,175,70).  visible(wk1gt_0,70). happensAt(walking(wk1gt_0),70). orientation(wk1gt_4,0,70). coords(wk1gt_4,36,200,70).  visible(wk1gt_4,70). happensAt(walking(wk1gt_4),70).
orientation(wk1gt_0,136,71). coords(wk1gt_0,219,173,71).  visible(wk1gt_0,71). happensAt(walking(wk1gt_0),71). orientation(wk1gt_4,0,71). coords(wk1gt_4,36,199,71).  visible(wk1gt_4,71). happensAt(walking(wk1gt_4),71).
orientation(wk1gt_0,136,72). coords(wk1gt_0,219,170,72).  visible(wk1gt_0,72). happensAt(walking(wk1gt_0),72). orientation(wk1gt_4,0,72). coords(wk1gt_4,36,199,72).  visible(wk1gt_4,72). happensAt(walking(wk1gt_4),72).
orientation(wk1gt_0,136,73). coords(wk1gt_0,219,168,73).  visible(wk1gt_0,73). happensAt(walking(wk1gt_0),73). orientation(wk1gt_4,0,73). coords(wk1gt_4,36,199,73).  visible(wk1gt_4,73). happensAt(walking(wk1gt_4),73).
orientation(wk1gt_0,136,74). coords(wk1gt_0,218,166,74).  visible(wk1gt_0,74). happensAt(walking(wk1gt_0),74). orientation(wk1gt_4,0,74). coords(wk1gt_4,36,199,74).  visible(wk1gt_4,74). happensAt(walking(wk1gt_4),74).
orientation(wk1gt_0,136,75). coords(wk1gt_0,218,162,75).  visible(wk1gt_0,75). happensAt(walking(wk1gt_0),75). orientation(wk1gt_4,0,75). coords(wk1gt_4,36,198,75).  visible(wk1gt_4,75). happensAt(walking(wk1gt_4),75).
orientation(wk1gt_0,136,76). coords(wk1gt_0,217,160,76).  visible(wk1gt_0,76). happensAt(walking(wk1gt_0),76). orientation(wk1gt_4,0,76). coords(wk1gt_4,36,198,76).  visible(wk1gt_4,76). happensAt(walking(wk1gt_4),76).
orientation(wk1gt_0,136,77). coords(wk1gt_0,215,156,77).  visible(wk1gt_0,77). happensAt(walking(wk1gt_0),77). orientation(wk1gt_4,0,77). coords(wk1gt_4,36,197,77).  visible(wk1gt_4,77). happensAt(walking(wk1gt_4),77).
orientation(wk1gt_0,131,78). coords(wk1gt_0,213,154,78).  visible(wk1gt_0,78). happensAt(walking(wk1gt_0),78). orientation(wk1gt_4,0,78). coords(wk1gt_4,37,197,78).  visible(wk1gt_4,78). happensAt(walking(wk1gt_4),78).
orientation(wk1gt_0,131,79). coords(wk1gt_0,212,153,79).  visible(wk1gt_0,79). happensAt(walking(wk1gt_0),79). orientation(wk1gt_4,0,79). coords(wk1gt_4,38,197,79).  visible(wk1gt_4,79). happensAt(walking(wk1gt_4),79).
orientation(wk1gt_0,124,80). coords(wk1gt_0,211,152,80).  visible(wk1gt_0,80). happensAt(walking(wk1gt_0),80). orientation(wk1gt_4,0,80). coords(wk1gt_4,38,196,80).  visible(wk1gt_4,80). happensAt(walking(wk1gt_4),80).
orientation(wk1gt_0,120,81). coords(wk1gt_0,210,151,81).  visible(wk1gt_0,81). happensAt(walking(wk1gt_0),81). orientation(wk1gt_4,0,81). coords(wk1gt_4,38,196,81).  visible(wk1gt_4,81). happensAt(walking(wk1gt_4),81).
orientation(wk1gt_0,115,82). coords(wk1gt_0,209,150,82).  visible(wk1gt_0,82). happensAt(walking(wk1gt_0),82). orientation(wk1gt_4,0,82). coords(wk1gt_4,38,196,82).  visible(wk1gt_4,82). happensAt(walking(wk1gt_4),82).
orientation(wk1gt_0,119,83). coords(wk1gt_0,209,150,83).  visible(wk1gt_0,83). happensAt(walking(wk1gt_0),83). orientation(wk1gt_4,0,83). coords(wk1gt_4,39,196,83).  visible(wk1gt_4,83). happensAt(walking(wk1gt_4),83).
orientation(wk1gt_0,121,84). coords(wk1gt_0,208,146,84).  visible(wk1gt_0,84). happensAt(walking(wk1gt_0),84). orientation(wk1gt_4,0,84). coords(wk1gt_4,39,196,84).  visible(wk1gt_4,84). happensAt(walking(wk1gt_4),84).
orientation(wk1gt_0,120,85). coords(wk1gt_0,208,144,85).  visible(wk1gt_0,85). happensAt(walking(wk1gt_0),85). orientation(wk1gt_4,0,85). coords(wk1gt_4,39,196,85).  visible(wk1gt_4,85). happensAt(walking(wk1gt_4),85).
orientation(wk1gt_0,119,86). coords(wk1gt_0,207,142,86).  visible(wk1gt_0,86). happensAt(walking(wk1gt_0),86). orientation(wk1gt_4,0,86). coords(wk1gt_4,39,195,86).  visible(wk1gt_4,86). happensAt(walking(wk1gt_4),86).
orientation(wk1gt_0,123,87). coords(wk1gt_0,207,138,87).  visible(wk1gt_0,87). happensAt(walking(wk1gt_0),87). orientation(wk1gt_4,0,87). coords(wk1gt_4,39,195,87).  visible(wk1gt_4,87). happensAt(walking(wk1gt_4),87).
#show holdsAt/2.