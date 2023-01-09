
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

	orientation(ricgt_3,51,39). coords(ricgt_3,301,286,39).  happensAt(appear(ricgt_3),39). happensAt(inactive(ricgt_3),39). orientation(ricgt_5,0,39). coords(ricgt_5,30,117,39).  visible(ricgt_5,39). happensAt(walking(ricgt_5),39).
orientation(ricgt_3,51,40). coords(ricgt_3,301,286,40).  visible(ricgt_3,40). happensAt(inactive(ricgt_3),40). orientation(ricgt_5,0,40). coords(ricgt_5,30,117,40).  visible(ricgt_5,40). happensAt(walking(ricgt_5),40).
orientation(ricgt_3,51,41). coords(ricgt_3,301,286,41).  visible(ricgt_3,41). happensAt(inactive(ricgt_3),41). orientation(ricgt_5,0,41). coords(ricgt_5,30,117,41).  visible(ricgt_5,41). happensAt(walking(ricgt_5),41).
orientation(ricgt_3,51,42). coords(ricgt_3,301,286,42).  visible(ricgt_3,42). happensAt(inactive(ricgt_3),42). orientation(ricgt_5,0,42). coords(ricgt_5,30,117,42).  visible(ricgt_5,42). happensAt(walking(ricgt_5),42).
orientation(ricgt_3,51,43). coords(ricgt_3,301,286,43).  visible(ricgt_3,43). happensAt(inactive(ricgt_3),43). orientation(ricgt_5,0,43). coords(ricgt_5,30,117,43).  visible(ricgt_5,43). happensAt(walking(ricgt_5),43).
orientation(ricgt_3,51,44). coords(ricgt_3,301,286,44).  visible(ricgt_3,44). happensAt(inactive(ricgt_3),44). orientation(ricgt_5,0,44). coords(ricgt_5,30,117,44).  visible(ricgt_5,44). happensAt(walking(ricgt_5),44).
orientation(ricgt_3,51,45). coords(ricgt_3,301,286,45).  visible(ricgt_3,45). happensAt(inactive(ricgt_3),45). orientation(ricgt_5,0,45). coords(ricgt_5,30,119,45).  visible(ricgt_5,45). happensAt(walking(ricgt_5),45).
orientation(ricgt_3,51,46). coords(ricgt_3,301,286,46).  visible(ricgt_3,46). happensAt(inactive(ricgt_3),46). orientation(ricgt_5,0,46). coords(ricgt_5,30,119,46).  visible(ricgt_5,46). happensAt(walking(ricgt_5),46).
orientation(ricgt_3,51,47). coords(ricgt_3,301,286,47).  visible(ricgt_3,47). happensAt(inactive(ricgt_3),47). orientation(ricgt_5,0,47). coords(ricgt_5,30,119,47).  visible(ricgt_5,47). happensAt(walking(ricgt_5),47).
orientation(ricgt_3,51,48). coords(ricgt_3,301,286,48).  visible(ricgt_3,48). happensAt(inactive(ricgt_3),48). orientation(ricgt_5,0,48). coords(ricgt_5,30,119,48).  visible(ricgt_5,48). happensAt(walking(ricgt_5),48).
orientation(ricgt_3,51,49). coords(ricgt_3,301,286,49).  visible(ricgt_3,49). happensAt(inactive(ricgt_3),49). orientation(ricgt_5,0,49). coords(ricgt_5,30,119,49).  visible(ricgt_5,49). happensAt(walking(ricgt_5),49).
orientation(ricgt_3,51,50). coords(ricgt_3,301,286,50).  visible(ricgt_3,50). happensAt(inactive(ricgt_3),50). orientation(ricgt_5,0,50). coords(ricgt_5,30,119,50).  visible(ricgt_5,50). happensAt(walking(ricgt_5),50).
orientation(ricgt_3,51,51). coords(ricgt_3,301,286,51).  visible(ricgt_3,51). happensAt(inactive(ricgt_3),51). orientation(ricgt_5,0,51). coords(ricgt_5,30,119,51).  visible(ricgt_5,51). happensAt(walking(ricgt_5),51).
orientation(ricgt_3,80,52). coords(ricgt_3,292,285,52).  visible(ricgt_3,52). happensAt(walking(ricgt_3),52). orientation(ricgt_5,0,52). coords(ricgt_5,30,119,52).  visible(ricgt_5,52). happensAt(walking(ricgt_5),52).
orientation(ricgt_3,80,53). coords(ricgt_3,290,283,53).  visible(ricgt_3,53). happensAt(walking(ricgt_3),53). orientation(ricgt_5,0,53). coords(ricgt_5,30,119,53).  visible(ricgt_5,53). happensAt(walking(ricgt_5),53).
orientation(ricgt_3,77,54). coords(ricgt_3,281,280,54).  visible(ricgt_3,54). happensAt(walking(ricgt_3),54). orientation(ricgt_5,0,54). coords(ricgt_5,29,120,54).  visible(ricgt_5,54). happensAt(walking(ricgt_5),54).
orientation(ricgt_3,70,55). coords(ricgt_3,280,279,55).  visible(ricgt_3,55). happensAt(walking(ricgt_3),55). orientation(ricgt_5,0,55). coords(ricgt_5,29,120,55).  visible(ricgt_5,55). happensAt(walking(ricgt_5),55).
orientation(ricgt_3,62,56). coords(ricgt_3,279,277,56).  visible(ricgt_3,56). happensAt(walking(ricgt_3),56). orientation(ricgt_5,0,56). coords(ricgt_5,29,121,56).  visible(ricgt_5,56). happensAt(walking(ricgt_5),56).
orientation(ricgt_3,58,57). coords(ricgt_3,278,277,57).  visible(ricgt_3,57). happensAt(walking(ricgt_3),57). orientation(ricgt_5,0,57). coords(ricgt_5,29,121,57).  visible(ricgt_5,57). happensAt(walking(ricgt_5),57).
orientation(ricgt_3,50,58). coords(ricgt_3,278,277,58).  visible(ricgt_3,58). happensAt(walking(ricgt_3),58). orientation(ricgt_5,0,58). coords(ricgt_5,29,121,58).  visible(ricgt_5,58). happensAt(walking(ricgt_5),58).
orientation(ricgt_3,50,59). coords(ricgt_3,277,276,59).  visible(ricgt_3,59). happensAt(walking(ricgt_3),59). orientation(ricgt_5,0,59). coords(ricgt_5,28,122,59).  visible(ricgt_5,59). happensAt(walking(ricgt_5),59).
orientation(ricgt_3,48,60). coords(ricgt_3,276,276,60).  visible(ricgt_3,60). happensAt(walking(ricgt_3),60). orientation(ricgt_5,0,60). coords(ricgt_5,28,123,60).  visible(ricgt_5,60). happensAt(walking(ricgt_5),60).
orientation(ricgt_3,45,61). coords(ricgt_3,272,276,61).  visible(ricgt_3,61). happensAt(walking(ricgt_3),61). orientation(ricgt_5,0,61). coords(ricgt_5,28,123,61).  visible(ricgt_5,61). happensAt(walking(ricgt_5),61).
orientation(ricgt_3,44,62). coords(ricgt_3,270,276,62).  visible(ricgt_3,62). happensAt(walking(ricgt_3),62). orientation(ricgt_5,0,62). coords(ricgt_5,28,123,62).  visible(ricgt_5,62). happensAt(walking(ricgt_5),62).
orientation(ricgt_3,40,63). coords(ricgt_3,265,276,63).  visible(ricgt_3,63). happensAt(walking(ricgt_3),63). orientation(ricgt_5,0,63). coords(ricgt_5,28,123,63).  visible(ricgt_5,63). happensAt(walking(ricgt_5),63).
#show holdsAt/2.